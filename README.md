# Gexture

Max MSP external for automatic k-shot gesture classification of MPU-6050 real-time measurement data.  
  
For usage, put the project folder into 'Max 8/Packages' and start Max.  
  
Build instructions for Win64:
- build the project in `build` using `cmake ..`
- build the `grt` third-party library the same way but in `third_party/grt/build/tmp` and then compile the .lib using VS2019 on the ALL_BUILD VS project file
- link to the built `grt.lib` in the main project following the steps described at the bottom of : https://github.com/nickgillian/grt/tree/master/build
- build the external using VS2019 on the Gexture VS project file in `build`.

### Debug
- MAX MSP external loading `error 126`: put a copy of `grt.dll` (compiled in the steps mentioned above) in `C:\Windows\System32`. Code 126 seems to be a windows dependency error (see: https://cycling74.com/forums/error-126-loading-external and https://github.com/worldmaking/Max_Worldmaking_Package/issues/23)
