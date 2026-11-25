#!/usr/bin/env ruby
# frozen_string_literal: true

# 大公共偏置（common bias / translation）下 Rumale 距离矩阵数值损失诊断
#
# 背景：
#   Rumale::PairwiseMetric#squared_error 采用展开式
#     ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y^T        （rumale-core/lib/rumale/pairwise_metric.rb:40-48）
#   再用 Numo::DFloat.maximum(err_mat, 0) 截断负数。
#   当所有坐标整体平移一个大常数 B 时，||x||^2、||y||^2、x.y^T 都达到 B^2 量级，
#   而真正承载距离信息的交叉项只有 B 量级，在浮点舍入中被吞掉，导致距离矩阵出现大量假零。
#
# 本脚本只依赖 numo-narray-alt 与仓库内的 rumale-core，可直接复跑：
#   ruby rumale-core/diagnostics/large_bias_distance_loss.rb
# 退出码 0 表示所有断言与实测一致；非 0 表示行为发生了变化（回归或修复）。

$LOAD_PATH.unshift(File.expand_path('../lib', __dir__))

require 'numo/narray'
require 'rumale/pairwise_metric'

PM = Rumale::PairwiseMetric
B = 1.0e10          # 整体平移的公共偏置
GAMMA = 1.0         # 题目指定的 RBF gamma
EXP_NEG1 = Math.exp(-1.0)

$failures = 0

def check(label, got, expected)
  ok = got == expected
  $failures += 1 unless ok
  tag = ok ? 'PASS' : 'FAIL'
  puts "  [#{tag}] #{label}: got=#{got.inspect} expected=#{expected.inspect}"
end

# 与 #squared_error 逐行同构、但不做 maximum(...,0) 截断的版本，
# 用于观察“截断前”的原始舍入结果。
def squared_error_raw(x, y = nil)
  sum_x_vec = (x**2).sum(axis: 1).expand_dims(1)
  sum_y_vec = y.nil? ? sum_x_vec.transpose : (y**2).sum(axis: 1).expand_dims(1).transpose
  yy = y.nil? ? x : y
  err = -2 * x.dot(yy.transpose)
  err += sum_x_vec
  err += sum_y_vec
  err
end

# 朴素直接相减的参考实现（先做差再平方，不经过展开式）。
def direct_squared(x, y)
  n_x = x.shape[0]
  n_y = y.shape[0]
  Numo::DFloat.cast(
    (0...n_x).to_a.product((0...n_y).to_a).map do |i, j|
      ((x[i, true] - y[j, true])**2).sum
    end
  ).reshape(n_x, n_y)
end

def show(title, mat, precision = 4)
  puts title
  mat.to_a.each do |row|
    puts '  ' + row.map { |v| sprintf("%#{precision + 8}.#{precision}f", v) }.join(' ')
  end
end

def line(c = '=')
  puts c * 78
end

line
puts 'Rumale 大公共偏置距离矩阵数值损失诊断'
puts "Ruby #{RUBY_VERSION}  B=#{B}  gamma=#{GAMMA}  Float::EPSILON=#{Float::EPSILON}"
line

# --------------------------------------------------------------------------
# 1. 无偏置基线：(0,0) 对 (1,0)，以及真重复点 (0,0) 对 (0,0)
# --------------------------------------------------------------------------
puts '1. 无偏置基线（展开式在小数量级下无损失）'
line('-')
x0 = Numo::DFloat[[0.0, 0.0]]
y0 = Numo::DFloat[[1.0, 0.0]]
d0 = Numo::DFloat[[0.0, 0.0]]   # 与 x0 完全重复的点

sq  = PM.squared_error(x0, y0)[0, 0]
euc = PM.euclidean_distance(x0, y0)[0, 0]
rbf = PM.rbf_kernel(x0, y0, GAMMA)[0, 0]
puts '  (0,0) vs (1,0)：理论平方距离=1，欧氏距离=1，RBF(gamma=1)=exp(-1)'
check('squared_error', sq, 1.0)
check('euclidean_distance', euc, 1.0)
check('rbf_kernel', rbf, EXP_NEG1)

sq  = PM.squared_error(x0, d0)[0, 0]
euc = PM.euclidean_distance(x0, d0)[0, 0]
rbf = PM.rbf_kernel(x0, d0, GAMMA)[0, 0]
puts '  (0,0) vs (0,0) 真重复点：理论距离 0/0，RBF=exp(0)=1'
check('squared_error (duplicate)', sq, 0.0)
check('euclidean_distance (duplicate)', euc, 0.0)
check('rbf_kernel (duplicate)', rbf, 1.0)
puts

# --------------------------------------------------------------------------
# 2. 整体平移 (10^10,10^10)，坐标差不变；定位最早的数值损失
# --------------------------------------------------------------------------
puts '2. 整体加 (10^10,10^10) 后：坐标差仍在，展开式在 B^2 量级首次损失'
line('-')
xb = Numo::DFloat[[B, 0.0]]
yb = Numo::DFloat[[B + 1.0, 0.0]]

diff = (yb - xb)[0, 0]
direct = direct_squared(xb, yb)[0, 0]
puts "  直接相减 yb-xb = #{diff.inspect}（坐标差 1 完整保留），朴素平方距离=#{direct}"
check('direct subtraction keeps delta=1', diff, 1.0)
check('direct squared distance', direct, 1.0)

sx  = (xb**2).sum(axis: 1)[0]
sy  = (yb**2).sum(axis: 1)[0]
dot = xb.dot(yb.transpose)[0, 0]
ulp = (B * B).next_float - (B * B)
puts '  展开式各分量（1e20 量级，1 ULP = 16384）：'
puts "    x.x  存储值 = #{sprintf('%.1f', sx)}  （整数真值 100000000000000000000）"
puts "    y.y  存储值 = #{sprintf('%.1f', sy)}  （整数真值 100000000020000000000，舍入误差 2048）"
puts "    x.y  存储值 = #{sprintf('%.1f', dot)}  （整数真值 100000000010000000000，舍入误差 7168）"
puts "    B^2 处 1 ULP = #{ulp}；承载距离的交叉项 2B = #{2 * B}（2B 本身约跨 #{(2 * B / ulp).round} 个 ULP，可表示）"
puts '    关键不在交叉项能否表示，而在三个 ~1e20 大项相减时，结果精度被大项 ULP=16384 锁死：'
puts '    各分量独立四舍五入的残差（本例 y.y 为 +2048、x.y 为 +7168）合并后达 16384，把真实平方距离 1 整个淹没'
puts '  => 最早损失发生在平方/点积这一步：分量一落到 B^2=1e20，ULP=16384，'
puts '     距离信息在三个大项加减合并时即被舍入吞掉，并非 sqrt 或 RBF 阶段。'

raw     = squared_error_raw(xb, yb)[0, 0]
clamped = PM.squared_error(xb, yb)[0, 0]
euc     = PM.euclidean_distance(xb, yb)[0, 0]
rbf     = PM.rbf_kernel(xb, yb, GAMMA)[0, 0]
puts '  同一对点 (B,0) vs (B+1,0)，三条流水线的最终输出：'
check('pre-clamp raw expanded value', raw, -16384.0)
check('squared_error (clamped)', clamped, 0.0)
check('euclidean_distance = sqrt(abs(...))', euc, 0.0)
check('rbf_kernel gamma=1', rbf, 1.0)
puts "    对照理论值：平方距离 1、欧氏距离 1、RBF #{EXP_NEG1.round(6)}（exp(-1)）"
puts

# --------------------------------------------------------------------------
# 3. 三类零 + 正误差：真零 / 截断前已舍入为零 / 负数被截断 / 正膨胀
# --------------------------------------------------------------------------
puts '3. 距离矩阵中“零”的三种成因（以 1D 点 (B,0) 与 (B+delta,0) 为代表）'
line('-')
puts '  (a) 真零：同一个点自身比较，截断前 raw 就恰好是 0.0'
raw_diag = squared_error_raw(xb)[0, 0]
check('self raw pre-clamp == 0.0', raw_diag, 0.0)
check('self squared_error == 0.0', PM.squared_error(xb)[0, 0], 0.0)
check('self rbf == 1.0', PM.rbf_kernel(xb, nil, GAMMA)[0, 0], 1.0)

puts '  (b) 截断前已舍入为零：delta=2（真平方距离 4），raw 恰好抵消成 0，不经过截断'
y2 = Numo::DFloat[[B + 2.0, 0.0]]
raw2 = squared_error_raw(xb, y2)[0, 0]
check('delta=2 pre-clamp raw == 0.0', raw2, 0.0)
check('delta=2 squared_error == 0.0', PM.squared_error(xb, y2)[0, 0], 0.0)
check('delta=2 rbf == 1.0 (假)', PM.rbf_kernel(xb, y2, GAMMA)[0, 0], 1.0)
puts "      理论 RBF=#{Math.exp(-4).round(6)}（exp(-4)），假零把它抬成 1.0"

puts '  (c) 负数被截断：delta=1（真平方距离 1），raw=-16384 被 maximum(raw,0) 截成 0'
check('delta=1 pre-clamp raw < 0', raw, -16384.0)
check('delta=1 clamped squared_error == 0.0', clamped, 0.0)
check('delta=1 rbf == 1.0 (假)', rbf, 1.0)
puts "      理论 RBF=#{EXP_NEG1.round(6)}（exp(-1)），截断把它抬成 1.0"

puts '  (d) 正向舍入膨胀：delta=4（真平方距离 16），raw=+16384，不截断但被放大 1024 倍'
y4 = Numo::DFloat[[B + 4.0, 0.0]]
raw4 = squared_error_raw(xb, y4)[0, 0]
rbf4 = PM.rbf_kernel(xb, y4, GAMMA)[0, 0]
check('delta=4 pre-clamp raw == 16384.0', raw4, 16384.0)
check('delta=4 euclidean == 128.0', PM.euclidean_distance(xb, y4)[0, 0], 128.0)
check('delta=4 rbf underflows to 0.0', rbf4, 0.0)
puts "      理论欧氏距离 4、RBF=#{Math.exp(-16).round(12)}（exp(-16)）；exp(-16384) 在 Float64 下溢为 0"
puts

# --------------------------------------------------------------------------
# 4. 显式 X/Y 与省略 Y 的自身比较；对角真零 vs 有偏置的非对角假零
# --------------------------------------------------------------------------
puts '4. 省略 Y 的自身比较 与 显式 X/Y：对角真零、非对角假零分别标注'
line('-')
# 1D 四点，偏移量分别取 0/1/2/4，恰好覆盖负截断、舍入零、正膨胀三类；无重复行
x_self = Numo::DFloat[[B, 0.0], [B + 1.0, 0.0], [B + 2.0, 0.0], [B + 4.0, 0.0]]
y_self = Numo::DFloat.cast(x_self.to_a)   # 数值相同的独立 Y 矩阵

raw_omit = squared_error_raw(x_self)
se_omit  = PM.squared_error(x_self)
se_expl  = PM.squared_error(x_self, y_self)
puts '  省略 Y（y=x 复用同一矩阵），截断前 raw（真值应为 delta^2：0/1/4/16）：'
show('  raw (pre-clamp):', raw_omit, 1)
show('  squared_error (clamped):', se_omit, 1)
show('  euclidean_distance:', PM.euclidean_distance(x_self), 4)
show('  rbf_kernel gamma=1:', PM.rbf_kernel(x_self, nil, GAMMA), 6)

diag_raw = raw_omit.diagonal
n4 = x_self.shape[0]
off_total = n4 * n4 - n4
off_false_zero = se_omit.eq(0).count - n4
off_neg = raw_omit.lt(0).count - raw_omit.diagonal.lt(0).count
off_round_zero = raw_omit.eq(0).count - n4
puts "  对角元素 raw = #{diag_raw.to_a.inspect}：全部恰好 0.0，是真零（同点自身比较），RBF 对角=1 正确"
check('diagonal raw all exactly zero', diag_raw.eq(0).all?, true)
puts "  非对角元素 #{off_total} 个：负数被截断 #{off_neg} 个、截断前已舍入为零 #{off_round_zero} 个，"
puts "  非对角假零合计 #{off_false_zero} 个；其余为正向膨胀（距离夸大、RBF 下溢为 0）"
check('off-diagonal negative-clamped count', off_neg, 4)
check('off-diagonal rounded-zero count', off_round_zero, 4)
check('off-diagonal false-zero count', off_false_zero, 8)
check('explicit Y matches omitted Y', se_expl.eq(se_omit).all?, true)
puts '  显式传入数值相同的独立 Y 矩阵，结果与省略 Y 完全一致（同一条展开式路径）'
puts
puts '  二维整体平移 (B,B) 再验：原始点 [[0,0],[1,0],[0,0]] 含 1 对真重复点，'
puts '  两个坐标的舍入残差恰好都抵消，raw 全矩阵为 0（非对角也全是假零，真重复点与假零无法区分）'
x2 = Numo::DFloat[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]] + B
raw2d = squared_error_raw(x2)
show('  2D raw (pre-clamp):', raw2d, 1)
show('  2D rbf_kernel gamma=1:', PM.rbf_kernel(x2, nil, GAMMA), 6)
check('2D off-diagonal false zeros', PM.squared_error(x2).eq(0).count - 3, 6)
check('2D explicit Y parity', PM.squared_error(x2, Numo::DFloat.cast(x2.to_a)).eq(PM.squared_error(x2)).all?, true)
puts

# --------------------------------------------------------------------------
# 5. 规模化普查：131 个偏置点，统计假零比例及其对 RBF 的影响
# --------------------------------------------------------------------------
puts '5. 普查：点 B..B+130 共 131 个（17030 个非对角点对）'
line('-')
z = Numo::DFloat.cast((0..130).map { |d| [B + d, 0.0] })
raw_z = squared_error_raw(z)
n = z.shape[0]
total_pairs = n * n
off_pairs = total_pairs - n
n_neg  = raw_z.lt(0).count - raw_z.diagonal.lt(0).count
n_zero = raw_z.eq(0).count - raw_z.diagonal.eq(0).count
n_pos  = off_pairs - n_neg - n_zero
clamped_zero = PM.squared_error(z).eq(0).count
puts "  非对角点对 #{off_pairs} 个："
puts "    截断前为负、被 maximum 截成 0 : #{n_neg}"
puts "    截断前已舍入为 0             : #{n_zero}"
puts "    假零合计                     : #{n_neg + n_zero}（占 #{(100.0 * (n_neg + n_zero) / off_pairs).round(2)}%）"
puts "    截断后矩阵零元素总数         : #{clamped_zero}（其中 #{n} 个是对角真零）"
puts "    正向膨胀（距离被夸大）点对   : #{n_pos}，RBF 普遍下溢成 0"
check('negative-before-clamp census', n_neg, 3050)
check('rounded-to-zero census', n_zero, 8146)
check('total zero entries census', clamped_zero, 11327)
puts

line
puts '结论汇总'
line
puts '  最早损失点：x**2 / x.dot(y^T) 把分量抬到 B^2=1e20 量级（该量级 1 ULP=16384），'
puts '             三个 ~1e20 大项相减时，真实距离（量级 1~16）被大项的舍入残差淹没，损失发生在加减合并阶段。'
puts '  三类零及 RBF(gamma=1) 影响：'
puts '    1) 真零（对角/真重复点）：raw 恰为 0，距离 0、RBF=1，语义正确。'
puts '    2) 截断前舍入为零（如 delta=2）：raw=0，距离报 0、RBF 报 1，'
puts '       不同点被当成同点（相似度被顶到上限 1.0）。'
puts '    3) 负数被截断（如 delta=1，raw=-16384）：maximum(raw,0) 抹掉负号，'
puts '       同样报距离 0、RBF=1；负数值本身永远到不了 sqrt（.abs 已是冗余）。'
puts '  附带的正向误差（如 delta=4，raw=16384 vs 真值 16）：距离夸大 1024 倍，'
puts '       RBF 从 exp(-16) 变成 exp(-16384)=0.0，相似度被压到下限。'
puts '  显式 Y 与省略 Y 走同一展开式，假零行为完全一致；只有直接相减的朴素路径'
puts '  （先 y-x 再平方）能保住坐标差 1。'
line

if $failures.zero?
  puts '全部断言通过：诊断结果可复现。'
  exit 0
else
  puts "#{$failures} 项断言失败：数值行为与诊断记录不符，请检查 Rumale 版本变化。"
  exit 1
end
