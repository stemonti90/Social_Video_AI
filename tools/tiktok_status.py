"""Ask TikTok what this account may actually post, refreshing the token first.

The one question that matters is whether PUBLIC_TO_EVERYONE is in privacy_level_options: without it
the pipeline can upload but never publish publicly.

It cannot tell you WHY it is missing, and do not try to infer it from the options that ARE offered.
Two independent gates remove PUBLIC_TO_EVERYONE and this endpoint reports neither of them:

  1. the app itself still being in review  -> developers.tiktok.com, the app's Production tab
  2. the TikTok ACCOUNT being private      -> the TikTok app, Settings > Privacy

Observed 2026-09-01 with the app in review AND the account private: the options came back as
FOLLOWER_OF_CREATOR / MUTUAL_FOLLOW_FRIENDS / SELF_ONLY. That looks like a private account's
signature, which is exactly why guessing from it is a trap — the review was pending too.

    PYTHONPATH=src .venv/bin/python tools/tiktok_status.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avp.config import Config          # noqa: E402
from avp.social import tokens as T     # noqa: E402
from avp.social.tiktok import TikTok   # noqa: E402


def main() -> int:
    rec = T.get("tiktok")
    if not rec:
        print("TikTok non collegato — esegui `avp connect tiktok`.")
        return 1
    tt = TikTok()
    if not T.is_fresh(rec):
        try:
            rec = tt.refresh(rec, Config.load("config.yaml"))
            T.put("tiktok", rec)
            print("Token rinnovato.")
        except Exception as e:  # noqa: BLE001 — report it, do not traceback at the user
            print(f"Rinnovo fallito ({e}) — riesegui `avp connect tiktok`.")
            return 1
    try:
        info = tt.creator_info(rec["access_token"])
    except Exception as e:  # noqa: BLE001
        print(f"creator_info fallito: {e}")
        return 1

    opts = info.get("privacy_level_options") or []
    print(f"Account   : {info.get('creator_nickname') or rec.get('account')}")
    print(f"Privacy   : {opts}")
    print(f"Duet/Stitch: disabilitati={info.get('duet_disabled')}/{info.get('stitch_disabled')}")
    if "PUBLIC_TO_EVERYONE" in opts:
        print("\nSBLOCCATO — la pipeline puo' pubblicare in pubblico.")
        return 0
    # Deliberately NOT guessing which gate is shut: both remove PUBLIC_TO_EVERYONE and this response
    # distinguishes neither. Name both places to look instead of naming a likely culprit.
    print("\nNON sbloccato. Servono DUE condizioni, entrambe indipendenti:")
    print("  1. app approvata  — developers.tiktok.com, scheda Production (non premere 'Recall')")
    print("  2. account non privato — app TikTok, Impostazioni > Privacy")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
