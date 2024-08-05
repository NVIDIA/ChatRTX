# Prerequisite
- Node and NPM installed

## Run ChatRTX - Development
- Set the python enviromnent path in file config.js at src\bridge_commands
- In one terminal run `npm run watch`
- In another run `npm run start-electron`

## Build ChatRTX
- Set the python enviromnent path in file config-packed.js at src\bridge_commands if needed.
- Run `npm run build-electron`. This creates dist and contains NVIDIA ChatRTX.exe at location dist\win-unpacked which launches ChatRTX app

Note: Above does not generate signed binaries, binaries needs to signed when app is distirbuted publicly.