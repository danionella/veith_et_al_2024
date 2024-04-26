import * as THREE from 'three';
import { DRACOLoader } from '../node_modules/three/examples/jsm/loaders/DRACOLoader.js';
import {GLTFLoader} from '../node_modules/three/examples/jsm/loaders/GLTFLoader.js';
import {OrbitControls} from '../node_modules/three/examples/jsm/controls/OrbitControls.js';

let camera, scene, renderer, controls;
const container =  document.createElement('div');
document.body.appendChild(container);

// Configure and create Draco decoder.
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath('../node_modules/three/examples/jsm/libs/draco/');
dracoLoader.setDecoderConfig({type: 'js'});

init();
animate();

function init() {
    // Camera
    camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 5);
    camera.position.set(0, .4, 0);
    camera.lookAt(0, 0, 0);

    // Scene
    scene = new THREE.Scene();
    //scene.add(new THREE.AxesHelper(5))
    scene.background = new THREE.Color(0xffffff);
    //scene.fog = new THREE.Fog(0x443333, 1, 4);

    // Fish
    const loader = new GLTFLoader();
    // // Optional: Provide a DRACOLoader instance to decode compressed mesh data
    loader.setDRACOLoader(dracoLoader);
    loader.load('./models/fish.gltf', function (gltf) {
        var model = gltf.scene;
        //var newMaterial = new THREE.MeshStandardMaterial({color: 0xff0000});
        model.traverse((o) => {
        if (o.isMesh) 
        //o.material = newMaterial;
        o.castShadow = true;
        o.receiveShadow = true;
        });

        gltf.scene.scale.set(0.02, 0.02, 0.02);

        const aabb = new THREE.Box3().setFromObject(gltf.scene);
        const centerHere = aabb.getCenter(new THREE.Vector3());
        gltf.scene.position.x += (gltf.scene.position.x - centerHere.x);
        gltf.scene.position.y += (gltf.scene.position.y - centerHere.y);
        gltf.scene.position.z += (gltf.scene.position.z - centerHere.z);

        gltf.scene.position.x += 0.12;
        gltf.scene.position.y += - 0.1;
        gltf.scene.position.z += - 0.01;
        console.log(gltf.scene.position)

        scene.add(gltf.scene);
        
        gltf.animations; // Array<THREE.AnimationClip>
        gltf.scene; // THREE.Group
        gltf.scenes; // Array<THREE.Group>
        gltf.cameras; // Array<THREE.Camera>
        gltf.asset; // Object
        },
        // called while loading is progressing
        function (xhr) {
            console.log((xhr.loaded / xhr.total * 100) + '% loaded');
        },
        // called when loading has errors
        function (error) {
            console.log('An error happened');
        }
    );

    // Lights
    //const hemiLight = new THREE.HemisphereLight(0x443333, 0x111122);
    //scene.add(hemiLight);
    //const ambientlight = new THREE.AmbientLight( 0x404040 ); // soft white light
    //scene.add(ambientlight);

    const spotLight = new THREE.SpotLight();
    spotLight.angle = Math.PI ;
    spotLight.penumbra = 0.5;
    spotLight.intensity = 1;
    spotLight.castShadow = true;
    spotLight.position.set(0, .6, 0);
    scene.add(spotLight);

    const spotLight2 = new THREE.SpotLight();
    spotLight2.angle = Math.PI ;
    spotLight2.penumbra = 0.5;
    spotLight2.intensity = 0.3;
    spotLight2.castShadow = true;
    spotLight2.position.set(0, -0.6, 0);
    scene.add(spotLight2);

    const spotLight3 = new THREE.SpotLight();
    spotLight3.angle = Math.PI ;
    spotLight3.penumbra = 0.5;
    spotLight3.intensity = 1;
    spotLight3.castShadow = true;
    spotLight3.position.set(0.1, 0, 0.6);
    scene.add(spotLight3);

    const spotLight4 = new THREE.SpotLight();
    spotLight4.angle = Math.PI ;
    spotLight4.penumbra = 0.5;
    spotLight4.intensity = 0.3;
    spotLight4.castShadow = true;
    spotLight4.position.set(-0.1, 0, -0.6);
    scene.add(spotLight4);

    // Renderer
    renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputEncoding = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // Orbit Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.listenToKeyEvents( window ); // optional
    //controls.addEventListener( 'change', render ); // call this only in static scenes (i.e., if there is no animation loop)
    //controls.enableDamping = true; // an animation loop is required when either damping or auto-rotation are enabled
    //controls.dampingFactor = 0.05;
    controls.screenSpacePanning = false;
    controls.minDistance = 0.01;
    controls.maxDistance = 3;
    controls.maxPolarAngle = Math.PI;

    window.addEventListener('resize', onWindowResize);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update(); //if using Orbit controls
    render();
}

function render() {
    ////turning scene:
    //const timer = Date.now() * 0.0003;
    //camera.position.z = Math.sin(timer) * .9;
    //camera.position.x = Math.cos(timer) * .9;
    //camera.lookAt(0, 0.1, 0);
    renderer.render(scene, camera);
}