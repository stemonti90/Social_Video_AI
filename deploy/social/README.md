# Publishing without Postiz

`avp` posts straight to TikTok, Instagram and YouTube. No broker, no extra services — the credentials
live in one 0600 file and the publishers are stateless.

## Why we left Postiz

Postiz works, but for one creator posting their own videos it means running Next.js + Postgres + Redis,
and — the part that actually cost us — it demands **six** TikTok scopes and its `checkScopes()` fails
authentication outright if any one of them is missing. Talking to TikTok directly needs **three**:
`user.info.basic`, `video.upload`, `video.publish`. TikTok's review explicitly delays apps that request
scopes they cannot demonstrate, so that difference is worth real time.

**What this does *not* avoid:** the platform approvals. TikTok's content-posting audit, Meta's app
review and Google's OAuth verification are imposed by the platforms, not by Postiz. Writing our own
client makes them *easier to pass*, not optional.

`publish.backend: postiz` still works if you want the other ~18 networks Postiz supports.

## One-time setup

### 1. App credentials

Prefer the environment — `config.yaml` is gitignored, but env vars keep secrets out of files entirely
and let the same checkout run against sandbox or production credentials.

```bash
export AVP_TIKTOK_CLIENT_KEY=…     AVP_TIKTOK_CLIENT_SECRET=…
export AVP_META_APP_ID=…           AVP_META_APP_SECRET=…
export AVP_GOOGLE_CLIENT_ID=…      AVP_GOOGLE_CLIENT_SECRET=…
```

The fallback is `publish.apps.<platform>.{client_id,client_secret}` in `config.yaml`.

### 2. Redirect URIs

Register exactly these in each platform's developer console:

| Platform  | Redirect URI |
|-----------|--------------|
| TikTok    | `https://www.astrostackerpro.com/connect/tiktok.html` |
| Instagram | `https://www.astrostackerpro.com/connect/instagram.html` |
| YouTube   | `https://www.astrostackerpro.com/connect/youtube.html` |

They are static pages on the main site. That is deliberate: TikTok and Meta both refuse plain-http
localhost redirects, and a real HTTPS callback service would mean a new subdomain, a new Cloudflare
tunnel route and a **second** domain verification — while `https://www.astrostackerpro.com/` is already
a verified URL prefix on TikTok. The pages make no network calls; they read the query string and print
the command to paste back.

### 3. Connect each account

```bash
PYTHONPATH=src .venv/bin/avp connect tiktok
```

Open the printed URL, authorise, and the callback page hands you the finishing command:

```bash
PYTHONPATH=src .venv/bin/avp connect tiktok --code '<CODE>'
```

Then check them any time:

```bash
PYTHONPATH=src .venv/bin/avp accounts
```

Roughly a once-a-year chore: TikTok refresh tokens last 365 days, Meta page tokens do not expire on
their own, and Google refreshes silently forever as long as the app stays authorised.

## Publishing

```bash
PYTHONPATH=src .venv/bin/avp publish <slug>        # DRY RUN → publish_plan.json
PYTHONPATH=src .venv/bin/avp publish <slug> --go   # post for real
```

A platform that fails is recorded in `publish_plan.json` with its error and the others still go out —
a broken TikTok upload is no reason to skip a good Instagram post.

## Things that will bite you

**Instagram fetches the video; it does not accept an upload.** `POST /{ig-user}/media` takes a
`video_url`, so the MP4 has to be publicly reachable for the length of one publish. `publish.media_host`
says where to stage it:

```yaml
media_host: {ssh: titan-prod, dir: /opt/astrostackerpro-site/media, url: "https://www.astrostackerpro.com/media"}
```

The file gets a random name and is deleted in a `finally`. `media/` is gitignored on the site — the site
deploys by `git pull` and a committed 20 MB video would sit in its history forever.

**Instagram needs a Business/Creator account linked to a Facebook Page.** A personal account cannot
publish through the API at all; no scope fixes that.

**YouTube caps unverified apps.** Until the Google Cloud project passes OAuth verification *and* a
YouTube API audit, every upload lands as **private** regardless of `privacyStatus`. The publisher
reports the privacy YouTube actually applied rather than the one requested, so this shows up in the log
instead of silently looking fine.

**TikTok answers a chunked upload with `206`.** That is success, not an error. Any client that treats
only `200 <= status < 300` … well, `206` is in that range — but a client checking `status == 200` will
report a perfectly good upload as failed.

**The AI-disclosure field is `is_aigc`.** `video_made_with_ai` is Postiz's internal name; TikTok ignores
it silently, and the video ships undisclosed. Controlled by `publish.disclose_ai`.
