let emotion_label;
var label_value;
var face;
var face_neutral;
var myCanvas;

let scoreThreshold = 0.5
let sizeType = '160'
let modelLoaded = false
var cImg;
var constraints = {
    audio: false,
    video: {
        width: 280,
        height: 180
    }
};
var EmotionModel;
var offset_x = 27;
var offset_y = 20;
var emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"];
var emotion_colors = ["#ff0000", "#00a800", "#ff4fc1", "#ffe100", "#306eff", "#ff9d00", "#7c7c7c"];

let forwardTimes = []

/* p5.js code */

function setup() {
    emotion_label = createDiv('neutral');
    emotion_label.parent('parent');
    emotion_label.addClass('emotion');

    myCanvas = createCanvas(1200, 800);
    myCanvas.parent('face');

    face = createSprite(340, 434, 200, 200);
    face_neutral = face.addAnimation('neutral', 'assets/neutral/1.png', 'assets/neutral/6.png', 'assets/neutral/7.png');
    face_neutral.offX = 600;
    face_neutral.offY = 400;
    face_neutral.scale = 0.1;
}

function draw() {
    emotion_label.html(label_value);
    drawSprites(face);
}


function updateTimeStats(timeInMs) {
    forwardTimes = [timeInMs].concat(forwardTimes).slice(0, 30)
    const avgTimeInMs = forwardTimes.reduce((total, t) => total + t) / forwardTimes.length
    $('#time').val(`${Math.round(avgTimeInMs)} ms`)
    $('#fps').val(`${faceapi.round(1000 / avgTimeInMs)}`)
}

function onIncreaseThreshold() {
    scoreThreshold = Math.min(faceapi.round(scoreThreshold + 0.1), 1.0)
    $('#scoreThreshold').val(scoreThreshold)
}

function onDecreaseThreshold() {
    scoreThreshold = Math.max(faceapi.round(scoreThreshold - 0.1), 0.1)
    $('#scoreThreshold').val(scoreThreshold)
}

function onSizeTypeChanged(e, c) {
    sizeType = e.target.value
    $('#sizeType').val(sizeType)
}

async function onPlay(videoEl) {
    if (videoEl.paused || videoEl.ended || !modelLoaded)
        return false

    const {
        width,
        height
    } = faceapi.getMediaDimensions(videoEl)
    const canvas = $('#overlay').get(0)
    canvas.width = width
    canvas.height = height

    const forwardParams = {
        inputSize: parseInt(sizeType),
        scoreThreshold
    }

    const ts = Date.now()
    const result = await faceapi.detectAllFaces(videoEl, new faceapi.TinyFaceDetectorOptions(forwardParams))
    console.result

    // console.log(result)
//            const result = await faceapi.tinyYolov2(videoEl, forwardParams)
    if (result.length != 0) {
        const context = canvas.getContext('2d')
        context.drawImage(videoEl, 0, 0, width, height)

        let ctx = context;
        ctx.lineWidth = 4;
        ctx.font = "25px Arial"
        ctx.fillText('Result', 0, 0);

        for (var i = 0; i < result.length; i++) {
            ctx.beginPath();
            var item = result[i].box;
            let s_x = Math.floor(item._x+offset_x);
            if (item.y<offset_y){
                var s_y = Math.floor(item._y);
            }
            else{
                var s_y = Math.floor(item._y-offset_y);
            }
            let s_w = Math.floor(item._width-offset_x);
            let s_h = Math.floor(item._height);
            let cT = ctx.getImageData(s_x, s_y, s_w, s_h);
            cT = preprocess(cT);

            z = EmotionModel.predict(cT)
            let index = z.argMax(1).dataSync()[0]
            let label = emotion_labels[index];

            label_value = emotion_labels[index];
            document.body.style.background = emotion_colors[index]

            // ctx.strokeStyle = emotion_colors[index];
            // ctx.rect(s_x, s_y, s_w, s_h);
            // ctx.stroke();
            // ctx.fillStyle = emotion_colors[index];
            // ctx.fillText(label, s_x, s_y);
            ctx.closePath();
        }

    }


    updateTimeStats(Date.now() - ts)

            //            faceapi.drawDetection('overlay', result.map(det => det.forSize(width, height)), {
            //                withScore: false
            //            })
    setTimeout(() => onPlay(videoEl))
    var status = document.getElementById('status');
    status.innerHTML = "";
}
async function loadNetWeights(uri) {
    return new Float32Array(await (await fetch(uri)).arrayBuffer())
}
// create model
async function createModel(path) {
    let model = await tf.loadLayersModel(path)
    return model
}
        // load emotion model
async function loadModel(path) {
            //            var lbl = document.getElementById("status");
            //            lbl.innerText = "Model Loading ..."
            //            let canvas = document.getElementById("combined");
            //            let cT = preprocess(cImg)
    EmotionModel = await createModel(path)
            //            z = model.predict(cT)
            //            toPixels(deprocess(z), canvas)
            //            lbl.innerText = "Model Loaded !"
}

function preprocess(imgData) {
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(imgData).toFloat();

        tensor = tensor.resizeBilinear([100, 100])

        tensor = tf.cast(tensor, 'float32')
        const offset = tf.scalar(255.0);
        // Normalize the image 
        const normalized = tensor.div(offset);
        //We add a dimension to get a batch shape 
        const batched = normalized.expandDims(0)
        return batched
    })
}

function successCallback(stream) {
    var videoEl = $('#inputVideo').get(0)
    videoEl.srcObject = stream;
}

function errorCallback(error) {
    alert(error)
    console.log("navigator.getUserMedia error: ", error);
    //            alert("navigator.getUserMedia error: ", error)
}

async function run() {
    const Model_url = 'model/tiny_face_detector/tiny_face_detector_model-weights_manifest.json'
    await faceapi.loadTinyFaceDetectorModel(Model_url)
    modelLoaded = true

    var status = document.getElementById('status');
    status.innerHTML = "Initializing the camera ... ";

    navigator.mediaDevices.getUserMedia(constraints)
        .then(successCallback)
        .catch(errorCallback);

    onPlay($('#inputVideo').get(0))
    $('#loader').hide()
}

$(document).ready(function() {
    loadModel('model/mobilenetv1_models/model.json')

    const sizeTypeSelect = $('#sizeType')
    sizeTypeSelect.val(sizeType)
    sizeTypeSelect.on('change', onSizeTypeChanged)
    sizeTypeSelect.material_select()
    run()
})

