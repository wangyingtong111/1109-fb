from datetime import timedelta
from unittest.mock import MagicMock

from django.contrib.auth.models import User
from django.db import models
from django.db.transaction import get_connection
from django.utils import timezone
import reversion
from reversion.models import Revision, Version
from test_app.models import TestModel, TestModelRelated, TestModelThrough, TestModelParent, TestMeta, TestModelInline
from test_app.tests.base import TestBase, TestBaseTransaction, TestModelMixin, UserMixin


class SaveTest(TestModelMixin, TestBase):

    def testModelSave(self):
        TestModel.objects.create()
        self.assertNoRevision()


class IsRegisteredTest(TestModelMixin, TestBase):

    def testIsRegistered(self):
        self.assertTrue(reversion.is_registered(TestModel))


class IsRegisterUnregisteredTest(TestBase):

    def testIsRegisteredFalse(self):
        self.assertFalse(reversion.is_registered(TestModel))


class GetRegisteredModelsTest(TestModelMixin, TestBase):

    def testGetRegisteredModels(self):
        self.assertEqual(set(reversion.get_registered_models()), {TestModel})


class RegisterTest(TestBase):

    def testRegister(self):
        reversion.register(TestModel)
        self.assertTrue(reversion.is_registered(TestModel))

    def testRegisterDecorator(self):
        @reversion.register()
        class TestModelDecorater(models.Model):
            pass
        self.assertTrue(reversion.is_registered(TestModelDecorater))

    def testRegisterAlreadyRegistered(self):
        reversion.register(TestModel)
        with self.assertRaises(reversion.RegistrationError):
            reversion.register(TestModel)

    def testRegisterM2MSThroughLazy(self):
        # When register is used as a decorator in models.py, lazy relations haven't had a chance to be resolved, so
        # will still be a string.
        @reversion.register()
        class TestModelLazy(models.Model):
            related = models.ManyToManyField(
                TestModelRelated,
                through="TestModelThroughLazy",
            )

        class TestModelThroughLazy(models.Model):
            pass


class UnregisterTest(TestModelMixin, TestBase):

    def testUnregister(self):
        reversion.unregister(TestModel)
        self.assertFalse(reversion.is_registered(TestModel))


class UnregisterUnregisteredTest(TestBase):

    def testUnregisterNotRegistered(self):
        with self.assertRaises(reversion.RegistrationError):
            reversion.unregister(User)


class CreateRevisionTest(TestModelMixin, TestBase):

    def testCreateRevision(self):
        with reversion.create_revision():
            obj = TestModel.objects.create()
        self.assertSingleRevision((obj,))

    def testCreateRevisionNested(self):
        with reversion.create_revision():
            with reversion.create_revision():
                obj = TestModel.objects.create()
        self.assertSingleRevision((obj,))

    def testCreateRevisionEmpty(self):
        with reversion.create_revision():
            pass
        self.assertNoRevision()

    def testCreateRevisionException(self):
        try:
            with reversion.create_revision():
                TestModel.objects.create()
                raise Exception("Boom!")
        except Exception:
            pass
        self.assertNoRevision()

    def testCreateRevisionDecorator(self):
        obj = reversion.create_revision()(TestModel.objects.create)()
        self.assertSingleRevision((obj,))

    def testPreRevisionCommitSignal(self):
        _callback = MagicMock()
        reversion.signals.pre_revision_commit.connect(_callback)

        with reversion.create_revision():
            TestModel.objects.create()
        self.assertEqual(_callback.call_count, 1)

    def testPostRevisionCommitSignal(self):
        _callback = MagicMock()
        reversion.signals.post_revision_commit.connect(_callback)

        with reversion.create_revision():
            TestModel.objects.create()
        self.assertEqual(_callback.call_count, 1)


class CreateRevisionAtomicTest(TestModelMixin, TestBaseTransaction):
    def testCreateRevisionAtomic(self):
        self.assertFalse(get_connection().in_atomic_block)
        with reversion.create_revision():
            self.assertTrue(get_connection().in_atomic_block)

    def testCreateRevisionNonAtomic(self):
        self.assertFalse(get_connection().in_atomic_block)
        with reversion.create_revision(atomic=False):
            self.assertFalse(get_connection().in_atomic_block)

    def testCreateRevisionInOnCommitHandler(self):
        from django.db import transaction
        from reversion.models import Revision

        self.assertEqual(Revision.objects.all().count(), 0)

        with reversion.create_revision(atomic=True):
            model = TestModel.objects.create()

            def on_commit():
                with reversion.create_revision(atomic=True):
                    model.name = 'oncommit'
                    model.save()

            transaction.on_commit(on_commit)

        self.assertEqual(Revision.objects.all().count(), 2)


class CreateRevisionManageManuallyTest(TestModelMixin, TestBase):

    def testCreateRevisionManageManually(self):
        with reversion.create_revision(manage_manually=True):
            TestModel.objects.create()
        self.assertNoRevision()

    def testCreateRevisionManageManuallyNested(self):
        with reversion.create_revision():
            with reversion.create_revision(manage_manually=True):
                TestModel.objects.create()
        self.assertNoRevision()


class CreateRevisionDbTest(TestModelMixin, TestBase):
    databases = {"default", "mysql", "postgres"}

    def testCreateRevisionMultiDb(self):
        with reversion.create_revision(using="mysql"), reversion.create_revision(using="postgres"):
            obj = TestModel.objects.create()
        self.assertNoRevision()
        self.assertSingleRevision((obj,), using="mysql")
        self.assertSingleRevision((obj,), using="postgres")


class CreateRevisionFollowTest(TestBase):

    def testCreateRevisionFollow(self):
        reversion.register(TestModel, follow=("related",))
        reversion.register(TestModelRelated)
        obj_related = TestModelRelated.objects.create()
        with reversion.create_revision():
            obj = TestModel.objects.create()
            obj.related.add(obj_related)
        self.assertSingleRevision((obj, obj_related))

    def testCreateRevisionFollowThrough(self):
        reversion.register(TestModel, follow=("related_through",))
        reversion.register(TestModelThrough, follow=("test_model", "test_model_related",))
        reversion.register(TestModelRelated)
        obj_related = TestModelRelated.objects.create()
        with reversion.create_revision():
            obj = TestModel.objects.create()
            obj_through = TestModelThrough.objects.create(
                test_model=obj,
                test_model_related=obj_related,
            )
        self.assertSingleRevision((obj, obj_through, obj_related))

    def testCreateRevisionFollowInvalid(self):
        reversion.register(TestModel, follow=("name",))
        with reversion.create_revision():
            with self.assertRaises(reversion.RegistrationError):
                TestModel.objects.create()


class CreateRevisionIgnoreDuplicatesTest(TestBase):

    def testCreateRevisionIgnoreDuplicates(self):
        reversion.register(TestModel, ignore_duplicates=True)
        with reversion.create_revision():
            obj = TestModel.objects.create()
        with reversion.create_revision():
            obj.save()
        self.assertSingleRevision((obj,))

    def testCreateRevisionIgnoreDuplicatesFollowRevertedDraft(self):
        # A followed object picked up implicitly with a draft value, then
        # explicitly saved back to its last stored value, must not leave a
        # stale draft version behind. Other genuinely changed objects in the
        # same revision are still versioned.
        reversion.register(TestModel, follow=("testmodelinline_set",), ignore_duplicates=True)
        reversion.register(TestModelInline, ignore_duplicates=True)
        with reversion.create_revision():
            obj = TestModel.objects.create(name="v1")
            inline_unchanged = TestModelInline.objects.create(test_model=obj, inline_name="v1")
            inline_changed = TestModelInline.objects.create(test_model=obj, inline_name="v1")
        with reversion.create_revision():
            obj.name = "v2"
            inline_unchanged.inline_name = "v2"
            inline_changed.inline_name = "v2"
            # Saving the parent follows its inlines, snapshotting the draft values.
            obj.save()
            # The first inline is explicitly reverted to its last stored value.
            inline_unchanged.inline_name = "v1"
            inline_unchanged.save()
            # The second inline really changes and is saved explicitly.
            inline_changed.save()
        # The revision only contains the parent and the genuinely changed inline.
        revision = Revision.objects.order_by("-pk").first()
        self.assertEqual(revision.version_set.count(), 2)
        self.assertTrue(
            Version.objects.get_for_object(obj).filter(revision=revision).exists()
        )
        self.assertFalse(
            Version.objects.get_for_object(inline_unchanged).filter(revision=revision).exists()
        )
        self.assertTrue(
            Version.objects.get_for_object(inline_changed).filter(revision=revision).exists()
        )
        # The reverted inline keeps only its original version.
        self.assertEqual(Version.objects.get_for_object(inline_unchanged).count(), 1)

    def testCreateRevisionIgnoreDuplicatesFollowRevertedDraftSharedChild(self):
        # The same followed object is reachable from two parents (a shared
        # child). Reverting its draft must only drop its own stale snapshot;
        # snapshots actually required by the revision are left untouched.
        reversion.register(TestModel, follow=("testmodelinline_set",), ignore_duplicates=True)
        reversion.register(TestModelInline, ignore_duplicates=True)
        with reversion.create_revision():
            obj_one = TestModel.objects.create(name="v1")
            obj_two = TestModel.objects.create(name="v1")
            shared_inline = TestModelInline.objects.create(test_model=obj_one, inline_name="v1")
            changed_inline = TestModelInline.objects.create(test_model=obj_two, inline_name="v1")
        with reversion.create_revision():
            obj_one.name = "v2"
            obj_two.name = "v2"
            shared_inline.inline_name = "v2"
            changed_inline.inline_name = "v2"
            # Both parents follow their inlines, snapshotting the draft values.
            obj_one.save()
            obj_two.save()
            # The shared child is explicitly reverted to its last stored value.
            shared_inline.test_model = obj_one
            shared_inline.inline_name = "v1"
            shared_inline.save()
            changed_inline.save()
        revision = Revision.objects.order_by("-pk").first()
        # Both parents and the genuinely changed inline are versioned; the
        # reverted shared child is not.
        self.assertEqual(revision.version_set.count(), 3)
        self.assertFalse(
            Version.objects.get_for_object(shared_inline).filter(revision=revision).exists()
        )
        self.assertTrue(
            Version.objects.get_for_object(changed_inline).filter(revision=revision).exists()
        )
        self.assertEqual(Version.objects.get_for_object(shared_inline).count(), 1)


class CreateRevisionInheritanceTest(TestModelMixin, TestBase):

    def testCreateRevisionInheritance(self):
        reversion.register(TestModelParent, follow=("testmodel_ptr",))
        with reversion.create_revision():
            obj = TestModelParent.objects.create()
        self.assertSingleRevision((obj, obj.testmodel_ptr))


class SetCommentTest(TestModelMixin, TestBase):

    def testSetComment(self):
        with reversion.create_revision():
            reversion.set_comment("comment v1")
            obj = TestModel.objects.create()
        self.assertSingleRevision((obj,), comment="comment v1")

    def testSetCommentNoBlock(self):
        with self.assertRaises(reversion.RevisionManagementError):
            reversion.set_comment("comment v1")


class GetCommentTest(TestBase):

    def testGetComment(self):
        with reversion.create_revision():
            reversion.set_comment("comment v1")
            self.assertEqual(reversion.get_comment(), "comment v1")

    def testGetCommentDefault(self):
        with reversion.create_revision():
            self.assertEqual(reversion.get_comment(), "")

    def testGetCommentNoBlock(self):
        with self.assertRaises(reversion.RevisionManagementError):
            reversion.get_comment()


class SetUserTest(UserMixin, TestModelMixin, TestBase):

    def testSetUser(self):
        with reversion.create_revision():
            reversion.set_user(self.user)
            obj = TestModel.objects.create()
        self.assertSingleRevision((obj,), user=self.user)

    def testSetUserNoBlock(self):
        with self.assertRaises(reversion.RevisionManagementError):
            reversion.set_user(self.user)


class GetUserTest(UserMixin, TestBase):

    def testGetUser(self):
        with reversion.create_revision():
            reversion.set_user(self.user)
            self.assertEqual(reversion.get_user(), self.user)

    def testGetUserDefault(self):
        with reversion.create_revision():
            self.assertEqual(reversion.get_user(), None)

    def testGetUserNoBlock(self):
        with self.assertRaises(reversion.RevisionManagementError):
            reversion.get_user()


class SetDateCreatedTest(TestModelMixin, TestBase):

    def testSetDateCreated(self):
        date_created = timezone.now() - timedelta(days=20)
        with reversion.create_revision():
            reversion.set_date_created(date_created)
            obj = TestModel.objects.create()
        self.assertSingleRevision((obj,), date_created=date_created)

    def testDateCreatedNoBlock(self):
        with self.assertRaises(reversion.RevisionManagementError):
            reversion.set_date_created(timezone.now())


class GetDateCreatedTest(TestBase):

    def testGetDateCreated(self):
        date_created = timezone.now() - timedelta(days=20)
        with reversion.create_revision():
            reversion.set_date_created(date_created)
            self.assertEqual(reversion.get_date_created(), date_created)

    def testGetDateCreatedDefault(self):
        with reversion.create_revision():
            self.assertAlmostEqual(reversion.get_date_created(), timezone.now(), delta=timedelta(seconds=1))

    def testGetDateCreatedNoBlock(self):
        with self.assertRaises(reversion.RevisionManagementError):
            reversion.get_date_created()


class AddMetaTest(TestModelMixin, TestBase):
    databases = {"default", "mysql", "postgres"}

    def testAddMeta(self):
        with reversion.create_revision():
            reversion.add_meta(TestMeta, name="meta v1")
            obj = TestModel.objects.create()
        self.assertSingleRevision((obj,), meta_names=("meta v1",))

    def testAddMetaNoBlock(self):
        with self.assertRaises(reversion.RevisionManagementError):
            reversion.add_meta(TestMeta, name="meta v1")

    def testAddMetaMultDb(self):
        with reversion.create_revision(using="mysql"), reversion.create_revision(using="postgres"):
            obj = TestModel.objects.create()
            reversion.add_meta(TestMeta, name="meta v1")
        self.assertNoRevision()
        self.assertSingleRevision((obj,), meta_names=("meta v1",), using="mysql")
        self.assertSingleRevision((obj,), meta_names=("meta v1",), using="postgres")
