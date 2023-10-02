const demosSection = document.getElementById('demos');

var model = undefined;

// Before we can use COCO-SSD class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
cocoSsd.load().then(function (loadedModel) {
  model = loadedModel;
  // Show demo section now model is ready to use.
  demosSection.classList.remove('invisible');
});


/********************************************************************
// Demo 1: Grab a bunch of images from the page and classify them
// upon click.
********************************************************************/

// In this demo, we have put all our clickable images in divs with the 
// CSS class 'classifyOnClick'. Lets get all the elements that have
// this class.
const imageContainers = document.getElementsByClassName('classifyOnClick');

// Now let's go through all of these and add a click event listener.
for (let i = 0; i < imageContainers.length; i++) {
  // Add event listener to the child element whichis the img element.
  imageContainers[i].children[0].addEventListener('click', handleClick);
}

// When an image is clicked, let's classify it and display results!
function handleClick(event) {
  if (!model) {
    console.log('Wait for model to load before clicking!');
    return;
  }
  
  // We can call model.classify as many times as we like with
  // different image data each time. This returns a promise
  // which we wait to complete and then call a function to
  // print out the results of the prediction.
  model.detect(event.target).then(function (predictions) {
    // Lets write the predictions to a new paragraph element and
    // add it to the DOM.
    console.log(predictions);
    for (let n = 0; n < predictions.length; n++) {
      // Description text
      const p = document.createElement('p');
      p.innerText = predictions[n].class  + ' - with ' 
          + Math.round(parseFloat(predictions[n].score) * 100) 
          + '% confidence.';
      // Positioned at the top left of the bounding box.
      // Height is whatever the text takes up.
      // Width subtracts text padding in CSS so fits perfectly.
      p.style = 'left: ' + predictions[n].bbox[0] + 'px;' + 
          'top: ' + predictions[n].bbox[1] + 'px; ' + 
          'width: ' + (predictions[n].bbox[2] - 10) + 'px;';

      const highlighter = document.createElement('div');
      highlighter.setAttribute('class', 'highlighter');
      highlighter.style = 'left: ' + predictions[n].bbox[0] + 'px;' +
          'top: ' + predictions[n].bbox[1] + 'px;' +
          'width: ' + predictions[n].bbox[2] + 'px;' +
          'height: ' + predictions[n].bbox[3] + 'px;';

      event.target.parentNode.appendChild(highlighter);
      event.target.parentNode.appendChild(p);
    }
  });
}



/********************************************************************
// Demo 2: Continuously grab image from webcam stream and classify it.
// Note: You must access the demo on https for this to work:
// https://tensorflow-js-image-classification.glitch.me/
********************************************************************/

const video = document.getElementById('webcam');
const liveView = document.getElementById('liveView');

// Check if webcam access is supported.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia);
}

// Keep a reference of all the child elements we create
// so we can remove them easilly on each render.
var children = [];


// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  const enableWebcamButton = document.getElementById('webcamButton');
  enableWebcamButton.addEventListener('click', enableCam);
} else {
  console.warn('getUserMedia() is not supported by your browser');
}


// Enable the live webcam view and start classification.
function enableCam(event) {
  if (!model) {
    console.log('Wait! Model not loaded yet.')
    return;
  }
  
  // Hide the button.
  event.target.classList.add('removed');  
  
  // getUsermedia parameters.
  const constraints = {
    video: true
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
    video.srcObject = stream;
    video.addEventListener('loadeddata', predictWebcam);
  });
}

let lastUploadedTime = 0; // Keeps track of the last time an image was uploaded.

function predictWebcam() {
  model.detect(video).then(function (predictions) {
    // Remove any highlighting we did in the previous frame.
    for (let i = 0; i < children.length; i++) {
      liveView.removeChild(children[i]);
    }
    children.splice(0);

    let highligher = null;
    // Loop through predictions and draw them to the live view if
    // they have a high confidence score.
    for (let n = 0; n < predictions.length; n++) {
      if (predictions[n].score > 0.66) {
        const p = document.createElement('p');
        p.innerText = predictions[n].class + ' - with '
          + Math.round(parseFloat(predictions[n].score) * 100)
          + '% confidence.';
        p.style = 'left: ' + predictions[n].bbox[0] + 'px;' +
          'top: ' + predictions[n].bbox[1] + 'px;' +
          'width: ' + (predictions[n].bbox[2] - 10) + 'px;';

        highlighter = document.createElement('div');
        highlighter.setAttribute('class', 'highlighter');
        highlighter.style = 'left: ' + predictions[n].bbox[0] + 'px; top: '
          + predictions[n].bbox[1] + 'px; width: '
          + predictions[n].bbox[2] + 'px; height: '
          + predictions[n].bbox[3] + 'px;';

        liveView.appendChild(highlighter);
        liveView.appendChild(p);

        children.push(highlighter);
        children.push(p);

        if (predictions[n].class.toLowerCase() === 'car' || predictions[n].class.toLowerCase() === 'vehicle') {

          console.log('car or vehicle detected');

          const [x, y, width, height] = predictions[n].bbox;

          // Get the ImageData for the bounding box from the excludeMaskCanvas
          const excludeCtx = excludeMaskCanvas.getContext('2d');
          const excludeImageData = excludeCtx.getImageData(x, y, width, height).data;

          // Check if there is any red pixel in the bounding box area of the excludeMaskCanvas
          let shouldExclude = false;
          for (let i = 0; i < excludeImageData.length; i += 4) {
            if (excludeImageData[i] === 255 && excludeImageData[i + 1] === 0 && excludeImageData[i + 2] === 0) { // Check for red pixels
              shouldExclude = true;
              break;
            }
          }

          if (shouldExclude) {
            highlighter.setAttribute('class', 'highlighter2');
            console.log('Skipping due to exclusion mask');
            continue; // Skip the current prediction due to exclusion mask
          }


          // Capture the current frame from the webcam.
          const canvas = document.createElement('canvas');
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;

          const ctx = canvas.getContext('2d');
          ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
          ctx.drawImage(excludeMaskCanvas, 0, 0);

          // Get the current time.
          const currentTime = new Date().getTime();

          // Check the time difference to manage the rate limit.
          if (currentTime - lastUploadedTime >= 2000) {
            lastUploadedTime = currentTime;
            canvas.toBlob(blob => {
              if (!blob) {
                console.error('Blob creation failed!');
                return;
              } 
              
              const imageUrl = URL.createObjectURL(blob);

              // Create an img element
              const imgElem = document.createElement('img');
              imgElem.src = imageUrl;

              // Append the img element to a div
              const debugDiv = document.getElementById('debugDiv'); // suppose debugDiv is the id of your div
              debugDiv.innerHTML = '';
              debugDiv.appendChild(imgElem);

              // ========================

              const formData = new FormData();
              formData.append('image', blob);

              const url = 'http://localhost:8010/proxy/v2/mmg/detect';
              const headers = new Headers({
                'api-key': '1191ff03-4fbf-4deb-a65d-faa3cf62566f',
                'accept': 'application/json',
              });

              fetch(url, {
                method: 'POST',
                headers: headers,
                body: formData,
              })
                .then(response => response.json())
                .then(data => {
                  if (data.is_success && data.detections.length > 0) {
                    for (const detection of data.detections) {
                      if (detection.mmg && detection.mmg.some(mmg => mmg.make_name.toLowerCase() === 'jaguar')) {
                        console.log("DETECTED");

                        let audio = new Audio('alarm.wav');
                        audio.play();

                      }
                    }
                  }
                })
                .catch(error => {
                  console.error('Error during API call', error);
                });
            }, 'image/jpeg');
          }
        }
      }
    }

    // Call this function again to keep predicting when the browser is ready.
    window.requestAnimationFrame(predictWebcam);
  });
}

const excludeMaskCanvas = document.getElementById('exclude_mask');

let ctxx = null;
let drawing = false;
let lastX = 0;
let lastY = 0;

video.addEventListener('loadeddata', function () {
  excludeMaskCanvas.width = video.videoWidth;
  excludeMaskCanvas.height = video.videoHeight;
  ctxx = excludeMaskCanvas.getContext('2d');
  ctxx.lineWidth = 50; // Set lineWidth to a reasonable value to observe changes

  excludeMaskCanvas.addEventListener('mousedown', (e) => {
    drawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
  });

  excludeMaskCanvas.addEventListener('mousemove', draw);
  excludeMaskCanvas.addEventListener('mouseup', () => drawing = false);
  excludeMaskCanvas.addEventListener('mouseout', () => drawing = false);
});

function draw(e) {
  if (!drawing) return;
  ctxx.strokeStyle = '#FF0000';
  ctxx.beginPath();
  ctxx.moveTo(lastX, lastY);
  ctxx.lineTo(e.offsetX, e.offsetY);
  ctxx.stroke();
  [lastX, lastY] = [e.offsetX, e.offsetY];
}
