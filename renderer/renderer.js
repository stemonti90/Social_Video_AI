document.getElementById("generateBtn").addEventListener("click", () => {
  window.api.send("generate-video", {});
});

window.api.receive("log", (msg) => {
  const out = document.getElementById("logOutput");
  out.textContent += `\n${msg}`;
});
