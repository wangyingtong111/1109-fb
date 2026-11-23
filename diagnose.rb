# encoding: utf-8
# diagnose.rb — pragmatic_segmenter 管线诊断
# 目标：把 清理前/后、列表保护、数值保护、最终边界 对应到原始行，
#       每个变化都标注实际入口（文件:行号 方法），不从结果反推。
$stdout.set_encoding(Encoding::UTF_8)
$LOAD_PATH.unshift File.expand_path('lib', __dir__)
require 'pragmatic_segmenter'

$TRACE = []

# ---------- 显示辅助 ----------
def esc(s) = s.to_s.gsub("\r", '␍').gsub("\n", '␊')

def show_change(before, after)
  b, a = esc(before), esc(after)
  if b.length <= 100 && a.length <= 100
    puts "      前: #{b}"
    puts "      后: #{a}"
  else
    i = 0
    i += 1 while i < before.length && i < after.length && before[i] == after[i]
    lo = [i - 15, 0].max
    puts "      前: …#{esc(before[lo, 70].to_s)}…"
    puts "      后: …#{esc(after[lo, 70].to_s)}…"
  end
end

def lib_caller
  loc = caller_locations(2, 14).find { |l| l.path.include?('pragmatic_segmenter') && !l.path.end_with?('diagnose.rb') }
  loc ? "#{File.basename(loc.path)}:#{loc.lineno} `#{loc.label}`" : '(外部)'
end

# ---------- 探针 1：所有 Rule.apply（规则级改写，带入口） ----------
module TraceRule
  def apply(str, *rules)
    rules.flatten.each do |rule|
      before = str.dup
      str.gsub!(rule.pattern, rule.replacement)
      next if str == before
      $TRACE << { kind: :rule, entry: lib_caller,
                  pattern: rule.pattern.inspect, repl: rule.replacement.inspect,
                  before: before, after: str.dup }
    end
    str
  end
end
PragmaticSegmenter::Rule.singleton_class.prepend(TraceRule)

# ---------- 探针 2：Cleaner 阶段与各私有步骤 ----------
module TraceCleaner
  def clean
    $TRACE << { kind: :head, name: "清理阶段 #{self.class}#clean", text: @text.dup }
    r = super
    $TRACE << { kind: :stage_end, name: '清理结束（→ 交给 Processor 的文本）', text: @text.dup }
    r
  end

  %i[remove_newline_in_middle_of_sentence remove_newline_in_middle_of_word
     replace_double_newlines replace_newlines remove_pdf_line_breaks
     check_for_no_space_in_between_sentences].each do |m|
    define_method(m) do
      before = @text.dup
      r = super()
      if @text != before
        $TRACE << { kind: :step, name: "Cleaner##{m}", before: before, after: @text.dup }
      end
      r
    end
  end
end
PragmaticSegmenter::Cleaner.prepend(TraceCleaner)

# ---------- 探针 3：List 列表保护 ----------
module TraceList
  def add_line_break
    $TRACE << { kind: :head, name: '列表保护 List#add_line_break', text: @text.dup }
    r = super
    $TRACE << { kind: :stage_end, name: '列表保护结束（♨ 已在此阶段内换成 ∯）', text: @text.dup }
    r
  end

  def scan_lists(regex1, regex2, replacement, strip: false)
    found = @text.scan(regex1).map(&:to_i)
    $TRACE << { kind: :note, name: "List#scan_lists（标记 #{replacement.inspect}）",
                detail: "扫描命中编号 #{found.inspect}；只有与相邻编号连续者才被采纳" }
    super
  end

  def substitute_found_list_items(regex, a, strip, replacement)
    before = @text.dup
    r = super
    if @text != before
      $TRACE << { kind: :step, name: "List#substitute_found_list_items：编号 #{a} 的句号 → #{a}#{replacement}",
                  before: before, after: @text.dup }
    end
    r
  end

  def add_line_breaks_for_numbered_list_with_periods
    fire = @text.include?('♨') && @text !~ /♨.+\n.+♨|♨.+\r.+♨/ && @text !~ /for\s\d{1,2}♨\s[a-z]/
    $TRACE << { kind: :note, name: 'List#add_line_breaks_for_numbered_list_with_periods',
                detail: "守卫 /♨.+\\n.+♨|♨.+\\r.+♨/ => #{fire ? '未命中，插入额外断行' : '命中：列表项已跨行，不再插断行'}" }
    super
  end
end
PragmaticSegmenter::List.prepend(TraceList)

# ---------- 探针 4：Processor 各保护阶段 ----------
module TraceProcessor
  LABEL = { replace_abbreviations: '缩写保护', replace_numbers: '数值保护',
            replace_continuous_punctuation: '连续标点保护',
            replace_periods_before_numeric_references: '数字引用边界' }.freeze

  def process(text:)
    $TRACE << { kind: :head, name: 'Processor#process（输入 = 清理后文本）', text: text.dup }
    super
  end

  %i[replace_abbreviations replace_numbers replace_continuous_punctuation
     replace_periods_before_numeric_references].each do |m|
    define_method(m) do
      before = @text.dup
      r = super()
      if @text != before
        $TRACE << { kind: :step, name: "Processor##{m}（#{LABEL[m]}）", before: before, after: @text.dup }
      end
      r
    end
  end

  def split_into_segments
    $TRACE << { kind: :head, name: '切分 split_into_segments：按 ␍ 硬切，再逐段恢复符号', text: @text.dup }
    r = super
    $TRACE << { kind: :final, segments: r }
    r
  end
end
PragmaticSegmenter::Processor.prepend(TraceProcessor)

# ---------- 探针 5：缩写替换器内部（直接 gsub!，不走 Rule.apply） ----------
module TraceAbbr
  %i[replace_prepositive_abbr replace_pre_number_abbr replace_period_of_abbr
     replace_multi_period_abbreviations].each do |m|
    define_method(m) do |txt, *rest|
      before = txt.dup
      r = super(txt, *rest)
      if r != before
        $TRACE << { kind: :step, name: "AbbreviationReplacer##{m}", before: before, after: r.dup }
      end
      r
    end
  end
end
PragmaticSegmenter::AbbreviationReplacer.prepend(TraceAbbr)

# ---------- 标记位置 → 原始行 ----------
MARKERS = { '♨' => '列表句号♨', '☝' => '列表括号☝', '∯' => '保护句号∯',
            'ȸ' => '临时句尾ȸ', 'ȹ' => '软换行ȹ', "\r" => '硬边界␍', "\n" => '换行␊' }.freeze

def line_of(str, i, anchors)
  fwd = str[i, 45].to_s
  bwd = str[[i - 45, 0].max, 45].to_s
  hit = anchors.find { |_, w| fwd.include?(w) } || anchors.find { |_, w| bwd.include?(w) }
  hit ? hit[0] : '??'
end

def marker_map(str, anchors)
  hits = []
  str.each_char.with_index do |ch, i|
    next unless MARKERS.key?(ch)
    ctx = esc(str[[i - 10, 0].max, 10].to_s) + '❰' + esc(ch) + '❱' + esc(str[i + 1, 12].to_s)
    hits << format('    @%-4d %-9s 属%s | %s', i, MARKERS[ch], line_of(str, i, anchors), ctx)
  end
  hits.empty? ? ['    （无保护标记 / 边界符）'] : hits
end

def print_event(ev, anchors)
  case ev[:kind]
  when :head
    puts "\n[阶段] #{ev[:name]}"
    puts "    文本: #{esc(ev[:text])}"
  when :stage_end
    puts "[阶段末] #{ev[:name]}"
    puts "    文本: #{esc(ev[:text])}"
    puts '    标记图:'; puts marker_map(ev[:text], anchors)
  when :step
    puts "  [改写] #{ev[:name]}"
    show_change(ev[:before], ev[:after])
  when :rule
    pat = ev[:pattern].length > 60 ? ev[:pattern][0, 60] + '…' : ev[:pattern]
    puts "  [规则] 入口 #{ev[:entry]}  #{pat} → #{ev[:repl]}"
    show_change(ev[:before], ev[:after])
  when :note
    puts "  [判定] #{ev[:name]} — #{ev[:detail]}"
  when :final
    puts "[阶段末] 切分完成，共 #{ev[:segments].length} 段"
  end
end

def report_segments(segs, anchors)
  seg_of = {}
  segs.each_with_index do |s, i|
    lines = anchors.select { |_, w| s.include?(w) }.map(&:first)
    tag = lines.empty? ? '（无锚点→碎片段）' : lines.join('+')
    puts "  S#{i + 1} [#{tag}] #{s.inspect}"
    anchors.each { |ln, w| seg_of[ln] = i if s.include?(w) }
  end
  cuts = {}
  puts '  边界判定（相邻原始行是否被切开）:'
  anchors.each_cons(2) do |(l1, _), (l2, _)|
    cut = !seg_of[l1].nil? && !seg_of[l2].nil? && seg_of[l1] != seg_of[l2]
    cuts["#{l1}|#{l2}"] = cut
    puts "    #{l1} | #{l2} : #{cut ? '有边界 ✓' : '无边界 ✗（落进同一句）'}"
  end
  cuts
end

$SUMMARY = []
def run_case(name, text, doc_type: nil, anchors:)
  $TRACE.clear
  puts "\n#{'=' * 88}\n案例 #{name}   doc_type=#{doc_type.inspect}   clean=true"
  puts '原始行:'
  text.split("\n").each_with_index { |l, i| puts "  L#{i + 1}: #{esc(l)}" }
  segs = PragmaticSegmenter::Segmenter.new(text: text, language: 'en', doc_type: doc_type).segment
  puts "\n-- 管线追踪（按真实执行顺序；规则改写均带入口）--"
  $TRACE.each { |ev| print_event(ev, anchors) }
  cleaned = $TRACE.find { |e| e[:kind] == :stage_end }[:text]
  puts "\n-- 最终分段 → 原始行映射 --"
  cuts = report_segments(segs, anchors)
  $SUMMARY << [name, esc(cleaned), segs, cuts]
  segs
end

A4 = [['L1', 'Read'], ['L2', 'First'], ['L3', 'Nested'], ['L4', 'Last']].freeze
BASE = "Read 1.2 ms.\n1. First item.\n  1.1 Nested item.\n2. Last item."

run_case('BASE（LF）', BASE, anchors: A4)
run_case('对照A：CRLF', BASE.gsub("\n", "\r\n"), anchors: A4)
run_case('对照B：嵌套项前插一个空行', "Read 1.2 ms.\n1. First item.\n\n  1.1 Nested item.\n2. Last item.", anchors: A4)
run_case('对照C：首行改为以 1.2 开头', "1.2 Read 1.2 ms.\n1. First item.\n  1.1 Nested item.\n2. Last item.", anchors: A4)

# ---- 最小区分例 ----
run_case('M1a：行首小节号 "1.2"（疑似被当测量小数）', '1.2 Introduction.', anchors: [['L1', 'Introduction']])
run_case('M1b：真测量小数 "1.2 ms"（对照）', 'Read 1.2 ms.', anchors: [['L1', 'Read']])
A2 = [['L1', 'Read'], ['L2', 'First']].freeze
run_case('M2a：PDF+行尾空格（疑似小节落进上一句）', "Read 1.2 ms. \n1. First item.", doc_type: 'pdf', anchors: A2)
run_case('M2b：PDF 无行尾空格（对照）', "Read 1.2 ms.\n1. First item.", doc_type: 'pdf', anchors: A2)

puts "\n#{'=' * 88}\n汇总表"
$SUMMARY.each do |name, cleaned, segs, cuts|
  c = cuts.map { |k, v| "#{k}:#{v ? 'Y' : 'N'}" }.join(' ')
  puts "  #{name}\n    清理后: #{cleaned}\n    段数=#{segs.length}  边界 #{c}"
end
