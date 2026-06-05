-- AUT Video Pipeline — double-clickable launcher (opens the Electron UI, no terminal).
-- Uses `open -n` so launchd owns the process and it survives this applet quitting.
do shell script "open -n '/Users/ste/Desktop/Progetti/AUT_VIDEO_PIPELINE/electron/node_modules/electron/dist/Electron.app' --args '/Users/ste/Desktop/Progetti/AUT_VIDEO_PIPELINE/electron'"
