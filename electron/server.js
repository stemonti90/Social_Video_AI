// Tiny static server for previewing the renderer in a browser (mock-data mode).
const http = require("http");
const fs = require("fs");
const path = require("path");

const dir = path.join(__dirname, "renderer");
const types = {
  ".html": "text/html", ".css": "text/css", ".js": "text/javascript",
  ".json": "application/json", ".png": "image/png", ".mp4": "video/mp4", ".svg": "image/svg+xml",
};

http.createServer((req, res) => {
  let p = decodeURIComponent((req.url || "/").split("?")[0]);
  if (p === "/") p = "/index.html";
  const file = path.join(dir, p);
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end("not found"); return; }
    res.writeHead(200, { "Content-Type": types[path.extname(file)] || "application/octet-stream" });
    res.end(data);
  });
}).listen(8770, () => console.log("avp-ui preview on http://localhost:8770"));
