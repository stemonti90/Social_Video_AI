// Secure bridge: the renderer can only call these whitelisted methods.
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("avp", {
  listProjects: () => ipcRenderer.invoke("avp:list"),
  getConfig: () => ipcRenderer.invoke("avp:config-get"),
  setConfig: (patch) => ipcRenderer.invoke("avp:config-set", patch),
  newProject: (topic) => ipcRenderer.invoke("avp:new", topic),
  deleteProject: (slug) => ipcRenderer.invoke("avp:delete", slug),
  readScript: (slug) => ipcRenderer.invoke("avp:readScript", slug),
  saveScript: (slug, text) => ipcRenderer.invoke("avp:saveScript", { slug, text }),
  readMetadata: (slug) => ipcRenderer.invoke("avp:readMetadata", slug),
  videoUrl: (slug, engine) => ipcRenderer.invoke("avp:videoUrl", { slug, engine }),
  publish: (slug, platforms, go) => ipcRenderer.invoke("avp:publish", { slug, platforms, go }),
  build: (slug) => ipcRenderer.invoke("avp:build", slug),
  onBuildEvent: (cb) => {
    const listener = (_e, ev) => cb(ev);
    ipcRenderer.on("avp:build-event", listener);
    return () => ipcRenderer.removeListener("avp:build-event", listener);
  },
  onNewEvent: (cb) => {
    const listener = (_e, ev) => cb(ev);
    ipcRenderer.on("avp:new-event", listener);
    return () => ipcRenderer.removeListener("avp:new-event", listener);
  },
});
