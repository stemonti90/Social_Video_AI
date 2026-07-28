"""Pure-logic tests for the control service — run with:  PYTHONPATH=server python -m unittest test_control"""
import json
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import control
import postiz_client as pz


class Slots(unittest.TestCase):
    def test_future_only_and_rollover(self):
        now = datetime(2026, 6, 29, 19, 30, tzinfo=ZoneInfo("Europe/Rome"))   # past 18:00, before 21:00
        slots = control.post_slots(now, ["12:00", "18:00", "21:00"], "Europe/Rome", 3)
        self.assertTrue(all(s > now for s in slots))
        self.assertEqual([(s.hour, s.day) for s in slots], [(21, 29), (12, 30), (18, 30)])
        self.assertTrue(control.iso_utc(slots[0]).endswith("Z"))


class PostizClientLogic(unittest.TestCase):
    META = {"youtube": {"title": "Cassini", "description": "d", "tags": ["space", "nasa"]},
            "tiktok": {"caption": "tok"}, "instagram": {"caption": "ig"}}

    def test_canon_and_settings(self):
        self.assertEqual(pz.canon("YT"), "youtube")
        tk = pz.settings_for("tiktok", self.META, disclose_ai=True, privacy="private")
        self.assertEqual(tk["__type"], "tiktok")
        self.assertEqual(tk["privacy_level"], "SELF_ONLY")
        self.assertTrue(tk["video_made_with_ai"])
        for k in ("duet", "stitch", "comment", "autoAddMusic", "content_posting_method"):
            self.assertIn(k, tk)
        yt = pz.settings_for("youtube", self.META, privacy="unlisted")
        self.assertEqual(yt["type"], "unlisted")
        self.assertEqual(yt["tags"][0], {"value": "space", "label": "space"})
        self.assertEqual(pz.settings_for("instagram", self.META), {"__type": "instagram", "post_type": "post"})

    def test_discover(self):
        class C:
            def list_integrations(self):
                return [{"id": "a", "identifier": "tiktok"}, {"id": "b", "identifier": "instagram"}]
        self.assertEqual(pz.discover(C()), {"tiktok": "a", "instagram": "b"})

    def test_create_post_body(self):
        cap = {}

        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b'{"id":"p1"}'

        def fake(req, timeout=None):
            cap["url"], cap["method"] = req.full_url, req.get_method()
            cap["auth"] = req.get_header("Authorization")
            cap["body"] = json.loads(req.data)
            return Resp()
        orig = pz.urllib.request.urlopen
        pz.urllib.request.urlopen = fake
        try:
            pz.PostizClient("http://x/api", "tok").create_post(
                "i1", "hi", {"id": "m"}, {"__type": "tiktok"}, "2026-07-01T18:00:00.000Z")
        finally:
            pz.urllib.request.urlopen = orig
        self.assertEqual(cap["auth"], "tok")
        self.assertTrue(cap["url"].endswith("/public/v1/posts"))
        self.assertEqual(cap["body"]["type"], "schedule")
        self.assertEqual(cap["body"]["date"], "2026-07-01T18:00:00.000Z")
        self.assertEqual(cap["body"]["posts"][0]["integration"]["id"], "i1")
        self.assertEqual(cap["body"]["posts"][0]["settings"], {"__type": "tiktok"})


class FakeClient:
    def __init__(self, connected):
        self.connected = connected
        self.posts = []

    def list_integrations(self):
        return [{"id": "i-" + p, "identifier": p} for p in self.connected]

    def upload(self, path):
        return {"id": "m1", "path": "http://x/m.mp4"}

    def create_post(self, iid, cap, media, settings, when, short_link=False):
        self.posts.append((iid, when, settings["__type"]))


class StoreLogic(unittest.TestCase):
    def _store(self, connected=("tiktok",)):
        fake = FakeClient(set(connected))
        s = control.Store(":memory:", client_factory=lambda: fake)
        return s, fake

    def test_topics_dedup_and_pop(self):
        s, _ = self._store()
        self.assertEqual(s.add_topics(["A", "B", "A", " "]), 2)     # dup + blank ignored
        self.assertEqual(s.topics(), ["A", "B"])
        self.assertEqual(s.pop_topics(1), ["A"])
        self.assertEqual(s.topics(), ["B"])

    def test_plan_creates_jobs_with_slots(self):
        s, _ = self._store()
        s.add_topics(["T1", "T2"])
        created = s.plan_day(count=2)
        self.assertEqual(len(created), 2)
        self.assertTrue(all(c["slot_utc"].endswith("Z") for c in created))
        self.assertEqual(s.status()["jobs"]["pending"], 2)
        self.assertEqual(s.status()["queue"], 0)                     # topics consumed

    def test_claim_is_atomic_pending_to_in_progress(self):
        s, _ = self._store()
        s.add_topics(["T1"])
        s.plan_day(count=1)
        j = s.claim("mac")
        self.assertIsNotNone(j)
        self.assertIsNone(s.claim("mac2"))                          # nothing left pending
        self.assertEqual(s.get(j["id"])["status"], "in_progress")

    def test_finalize_posts_to_connected_only(self):
        control.CFG["platforms"] = ["tiktok", "instagram"]
        s, fake = self._store(connected=("tiktok",))               # instagram NOT connected
        s.add_topics(["T1"])
        jid = s.plan_day(count=1)[0]["id"]
        s.claim("mac")
        s.save_meta(jid, json.dumps({"tiktok": {"caption": "c"}, "youtube": {"title": "t"}}))
        res = s.finalize(jid, "/tmp/nonexistent.mp4")
        self.assertEqual(res["status"], "done")
        self.assertEqual(res["posted_to"], ["tiktok"])
        self.assertEqual([p[2] for p in fake.posts], ["tiktok"])   # one post, tiktok only

    def test_finalize_generate_only_when_no_channel(self):
        control.CFG["platforms"] = ["tiktok", "instagram"]
        s, fake = self._store(connected=())                        # nothing connected
        s.add_topics(["T1"])
        jid = s.plan_day(count=1)[0]["id"]
        res = s.finalize(jid, "/tmp/nonexistent.mp4")
        self.assertEqual(res["status"], "done")
        self.assertEqual(res["posted_to"], [])                     # kept, not posted
        self.assertEqual(fake.posts, [])


if __name__ == "__main__":
    unittest.main()
