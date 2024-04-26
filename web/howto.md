# 3D viewer
visit: 

## How to render your 3D mesh with three.js

### 1. Get the example running
- install nodejs  
https://nodejs.org/en/download/  

- set up a project ("fish") with a javascript bundler

      mkdir fish
      cd fish
      npm init -y
      npm install webpack webpack-cli --save-dev

-  open the project in VSCode

-  add these folders and files

        /dist
        |__ index.html
        |__ /models
        |__ |__ fish.gltf
        |__ |__ bunny.drc
        /src
        |__ index.js
        
- install three

      npm install --save three
          
    This adds three to the `node_modules` folder.
     
- add build command into `package.json`  
https://webpack.js.org/guides/getting-started/#npm-scripts   

- build

      npm run build

- run a local server  
    for this, install the live server in VSCode:

      Open VSCode and type ctrl+P, type ext install ritwickdey.liveserver.

    right-click on index.html -> Open with Live Server

- in the browser
    checkout the console for debugging:
    right-click -> Inspect 
    -> Console
    
### 2. Render your own 3D graphic

- Export `model.glb` with Blender (you can export a selection only, use decimating modifiers rather than compression)  
- Compress the file with draco

        npm install -g gltf-pipeline
        gltf-pipeline -i model.glb -o modelDraco.gltf -d
    More info: https://github.com/CesiumGS/gltf-pipeline  
- Copy it to `dist/models/` and adapt paths in `src/index.js`

### Resources
- Packaging  
https://webpack.js.org/guides/getting-started/  
- Three.js documentation
https://threejs.org/docs/index.html#manual/en/introduction/Installation  
- Three.js examples  
View https://threejs.org/examples/#webgl_animation_keyframes  
Code https://github.com/mrdoob/three.js/tree/dev/examples  