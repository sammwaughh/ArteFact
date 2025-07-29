// ==========================
// == GLOBAL CONFIGURATION ==
// ==========================
const API_BASE_URL = 'http://127.0.0.1:8000';

// Variables to store session/run state
let runId;
let s3Key;
let upload;

// --- Available models list ---
let availableModels = [];
let selectedModel = '';

/**
 * Appends a message to the working log with the specified type.
 * @param {string} message - The message to display.
 * @param {string} [type='text-white'] - The CSS class for the message (e.g., 'text-white', 'text-danger').
 */
function logWorkingMessage(message, type = 'text-white') {
  const logContainer = $('#workingLog');
  logContainer.append(`<div class="${type}">${message}</div>`);
  logContainer.scrollTop(logContainer[0].scrollHeight);
}

// ==========================
// == TOPIC TAG SELECTION  ==
// ==========================
let selectedTopics = [];
let topicMap = {};

/**
 * Updates the display of selected topics in the #selectedTopicsWrapper.
 * Shows all topics from topicMap, visually indicating which are selected.
 */
function updateSelectedTopicsDisplay() {
  $('#selectedTopicsWrapper').removeClass('d-none');
  const selectedTagContainer = $('#selectedTopicTags');
  selectedTagContainer.empty();

  for (const [code, label] of Object.entries(topicMap)) {
    const isSelected = selectedTopics.includes(code);
    const tag = $('<button>')
      .addClass('btn btn-sm px-3 py-1 rounded-pill')
      .addClass(isSelected ? 'btn-primary' : 'btn-outline-secondary')
      .text(label)
      .data('code', code)
      .on('click', function () {
        const idx = selectedTopics.indexOf(code);
        if (idx === -1) {
          selectedTopics.push(code);
        } else {
          selectedTopics.splice(idx, 1);
        }
        updateSelectedTopicsDisplay();
        $(`#topicTags button[data-code="${code}"]`)
          .toggleClass('active')
          .toggleClass('btn-primary')
          .toggleClass('btn-outline-primary');
      });
    selectedTagContainer.append(tag);
  }
}

// Main script entry point: sets up event handlers on document ready
$(document).ready(function () {
  // --- Load topic tags from /topics ---
  fetch(`${API_BASE_URL}/topics`)
    .then(response => response.json())
    .then(data => {
      topicMap = data;
      const tagContainer = document.getElementById('topicTags');
      for (const [code, label] of Object.entries(data)) {
        const tag = document.createElement('button');
        tag.className = 'btn btn-outline-primary btn-sm px-3 py-1 rounded-pill';
        tag.textContent = label;
        tag.dataset.code = code;
        tag.addEventListener('click', function () {
          this.classList.toggle('active');
          if (this.classList.contains('active')) {
            this.classList.replace('btn-outline-primary', 'btn-primary');
            selectedTopics.push(code);
          } else {
            this.classList.replace('btn-primary', 'btn-outline-primary');
            selectedTopics = selectedTopics.filter(c => c !== code);
          }
        });
        tagContainer.appendChild(tag);
      }
    })
    .catch(error => {
      console.error('Error loading topics:', error);
    });

  // --- Load model list from /models ---
  fetch(`${API_BASE_URL}/models`)
    .then(response => response.json())
    .then(data => {
      availableModels = data;
      console.log("Available models:", availableModels);
      // Populate the model dropdown
      const dropdownMenu = $('#modelDropdownMenu');
      if (availableModels.length > 0) {
        dropdownMenu.empty();
        availableModels.forEach((model, index) => {
          const item = $('<li><a class="dropdown-item" href="#">' + model + '</a></li>');
          if (index === 0) {
            $('#modelDropdown').text('Model: ' + model);
            item.find('a').addClass('active');
            selectedModel = model;
          }
          item.on('click', function () {
            selectedModel = model;
            $('#modelDropdownMenu a').removeClass('active');
            $(this).find('a').addClass('active');
            $('#modelDropdown').text('Model: ' + model);
          });
          dropdownMenu.append(item);
        });
      }
    })
    .catch(error => {
      console.error('Error loading models:', error);
    });
  // Trigger file upload dialog when upload button is clicked
  $('#uploadTrigger').on('click', function () {
    $('#imageUpload').click();
  });

  // Handle image file selected from file input
  $('#imageUpload').on('change', function (event) {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = function (e) {
        $('#uploadedImage').attr('src', e.target.result).removeClass('d-none');
        $('#uploadTrigger').addClass('d-none');
        $('.card:has(#uploadTrigger)').remove();
        $('#exampleContainer').remove();
        $('#workingOverlay').removeClass('d-none');
        $('#imageTools').removeClass('d-none');
        fetchPresign();
      };
      reader.readAsDataURL(file);
    }
  });

  // --- Drag and drop support for uploading images ---
  $('#uploadedImageContainer').on('dragover', function (e) {
    e.preventDefault();
    e.stopPropagation();
    $(this).addClass('border border-light');
  });

  $('#uploadedImageContainer').on('dragleave', function (e) {
    e.preventDefault();
    e.stopPropagation();
    $(this).removeClass('border border-light');
  });

  $('#uploadedImageContainer').on('drop', function (e) {
    e.preventDefault();
    e.stopPropagation();
    $(this).removeClass('border border-light');
    const file = e.originalEvent.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = function (e) {
        $('#uploadedImage').attr('src', e.target.result).removeClass('d-none');
        $('#uploadTrigger').addClass('d-none');
        $('#workingOverlay').removeClass('d-none');
        $('#imageTools').removeClass('d-none');
        fetchPresign();
      };
      reader.readAsDataURL(file);
    }
  });

  // --- Example image selection logic ---
  let selectedSrc = null;
  $('.example-img').on('click', function () {
    $('.example-img').removeClass('selected-img');
    $(this).addClass('selected-img');
    selectedSrc = $(this).attr('src');
    $('#selectImageBtn').removeClass('d-none');
  });

  // Handle selection of an example image
  $('#selectImageBtn').on('click', function () {
    if (selectedSrc) {
      $('#uploadedImage').attr('src', selectedSrc).removeClass('d-none');
      $('#uploadTrigger').addClass('d-none');
      $('.card:has(#uploadTrigger)').remove();
      $('#exampleContainer').remove();
      $('#workingOverlay').removeClass('d-none');
      $('#imageTools').removeClass('d-none');
      fetchPresign();
    }
  });

  // Toggle debug panel visibility
  $('#debugPanelToggle button').on('click', function () {
    $('#debugPanel').toggle();
  });
});

/**
 * Initiates a new session by requesting a presigned upload URL
 * and registering a run. Triggers polling for run status.
 */
function fetchPresign() {
  $('#debugStatus').text('Requesting ID...');
  logWorkingMessage('Requesting Session ID...', 'text-white');

  fetch(`${API_BASE_URL}/presign`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ fileName: 'selected.jpg' })
  })
    .then(res => res.json())
    .then(data => {
      runId = data.runId;
      s3Key = data.s3Key;
      upload = data.upload;

      // --- Unified canvas-based image upload logic ---
      const imgElement = document.getElementById('uploadedImage');
      const canvas = document.createElement('canvas');
      canvas.width = imgElement.naturalWidth;
      canvas.height = imgElement.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(imgElement, 0, 0);

      canvas.toBlob(function (blob) {
        const file = new File([blob], 'uploaded.jpg', { type: blob.type });
        const formData = new FormData();
        formData.append('file', file);

        fetch(`${API_BASE_URL}/upload/${runId}`, {
          method: 'POST',
          body: formData
        })
        .then(res => {
          if (res.status === 204) {
            logWorkingMessage('Image uploaded successfully (204 No Content)', 'text-white');
          } else {
            return res.json().then(() => {
              logWorkingMessage('Image uploaded successfully', 'text-white');
            });
          }
        })
        .then(() => {
          $('#debugSessionId').text(runId);
          $('#debugStatus').text('Got ID');
          logWorkingMessage('Session ID: ' + runId, 'text-white');
          logWorkingMessage('Sending /runs request...', 'text-white');
          $('#debugStatus').text('Posting run info');

          // Show selected topics box using helper
          updateSelectedTopicsDisplay();

          return fetch(`${API_BASE_URL}/runs`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              runId: runId,
              s3Key: s3Key,
              topics: selectedTopics,
              model: selectedModel
            })
          });
        })
        .then(res => {
           // Don't parse empty 202 response as JSON
           if (res.status === 202) {
             return {};
           }
           return res.json();
         })
         .then(response => {
          logWorkingMessage('Run registered successfully', 'text-white');
          $('#debugStatus').text('Run submitted');
          pollRunStatus(runId);
         })
        .catch(err => {
          console.error('Upload or /runs error:', err);
          logWorkingMessage('Error uploading image or submitting run', 'text-danger');
          $('#debugStatus').text('Run submission failed');
        });
      }, 'image/jpeg');
      // --- End unified image upload logic ---
    })
    .catch(err => {
      console.error('Presign error:', err);
      $('#debugStatus').text('Error fetching ID');
      logWorkingMessage('Error fetching ID', 'text-danger');
    });
}

/**
 * Polls the backend for the status of the current run.
 * When complete, fetches and displays output sentences.
 * @param {string} runId - The run/session ID to poll.
 */
function pollRunStatus(runId) {
  logWorkingMessage('Polling run status...', 'text-white');

  const intervalId = setInterval(() => {
    fetch(`${API_BASE_URL}/runs/${runId}`)
      .then(res => res.json())
      .then(data => {
        $('#debugStatus').text(`Status: ${data.status}`);
        logWorkingMessage(`Status: ${data.status}`, 'text-white');

        if (data.status !== 'processing') {
          clearInterval(intervalId);
          logWorkingMessage('Processing complete', 'text-white');

          if (data.status === 'done') {
             // Use outputKey from backend response instead of deriving from upload URL
             const filePath = data.outputKey ? data.outputKey.split('/').pop() : `${runId}.json`;
             
             logWorkingMessage('Fetching outputs from: ' + filePath, 'text-white');

            // Fetch output sentences from backend
            fetch(`${API_BASE_URL}/outputs/${filePath}`)
              .then(res => res.json())
              .then(output => {
                logWorkingMessage('Outputs received', 'text-white');
                display_sentences(output);
                $('#workingOverlay').addClass('d-none');
              })
              .catch(err => {
                console.error('Error fetching outputs:', err);
                logWorkingMessage('Error fetching outputs', 'text-danger');
              });
          }
          else if (data.status === 'error') {
            logWorkingMessage('An error occurred during processing.', 'text-danger');
            setTimeout(() => {
              $('#workingOverlay').addClass('d-none');
              // $('#uploadedImage').addClass('d-none').attr('src', '');
              // $('#uploadTrigger').removeClass('d-none');
              // $('#imageTools').addClass('d-none');
              // $('.col-md-3').addClass('d-none');
              // $('#imageHistoryWrapper').addClass('d-none');
              // $('#selectedTopicsWrapper').addClass('d-none');
              $('#debugStatus').text('Idle');
              $('#debugSessionId').text('N/A');
              selectedTopics = [];
              // $('#topicTags button').removeClass('active btn-primary').addClass('btn-outline-primary');
              // $('#selectedTopicTags').empty();
            }, 5000);
          }
        }
      })
      .catch(err => {
        console.error('Polling error:', err);
        logWorkingMessage('Error polling status', 'text-danger');
        clearInterval(intervalId);
      });
  }, 1000);
}

/**
 * Escapes HTML special characters in a string to prevent XSS.
 * @param {string} str - The string to escape.
 * @returns {string}
 */
function escapeHTML(str) {
  return str.replace(/[&<>'"]/g, tag => (
    {'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'}[tag]
  ));
}

/**
 * Displays the list of output sentences in the sidebar.
 * @param {Array} data - Array of sentence objects.
 */
function display_sentences(data) {
  // Show the sentences panel
  $('.col-md-3').removeClass('d-none');
   $('#sentenceList').empty();

  data.forEach((item, index) => {
    const sentenceItem = $(`
      <li class="list-group-item">
        <div class="d-flex">
          <div class="flex-grow-1">
            <p class="mb-1">${escapeHTML(item["english_original"])}
            <button class="btn btn-sm btn-outline-primary" onclick="lookupDOI('${item["work"]}')" title="Look Up DOI">
              <i class="bi bi-search"></i>
            </button>
            </p>
          </div>
        </div>
      </li>
    `);
    $('#sentenceList').append(sentenceItem);
  });
}

// --- Begin Crop Tool Functionality ---
// Variables for cropping state
let isCropping = false;
let cropStartX = 0;
let cropStartY = 0;
let cropRect = null;

// Activate cropping mode when crop tool button is clicked
$('#cropToolBtn').on('click', function () {
  isCropping = true;
  $('#uploadedImageContainer').css('cursor', 'crosshair');
});

// Start drawing crop rectangle on mouse down
$('#uploadedImageContainer').on('mousedown', function (e) {
  if (!isCropping) return;
  const rect = this.getBoundingClientRect();
  cropStartX = e.clientX - rect.left;
  cropStartY = e.clientY - rect.top;

  if (cropRect) {
    cropRect.remove();
  }

  cropRect = $('<div>')
    .addClass('position-absolute border border-warning')
    .css({
      left: cropStartX,
      top: cropStartY,
      width: 0,
      height: 0,
      zIndex: 10,
      pointerEvents: 'none'
    })
    .appendTo('#uploadedImageContainer');
});

// Update crop rectangle size on mouse move
$('#uploadedImageContainer').on('mousemove', function (e) {
  if (!isCropping || !cropRect) return;
  const rect = this.getBoundingClientRect();
  const currentX = e.clientX - rect.left;
  const currentY = e.clientY - rect.top;

  const width = Math.abs(currentX - cropStartX);
  const height = Math.abs(currentY - cropStartY);
  const left = Math.min(currentX, cropStartX);
  const top = Math.min(currentY, cropStartY);

  cropRect.css({ left, top, width, height });
});

// Complete cropping on mouse up, update image, and save history
$('#uploadedImageContainer').on('mouseup', function (e) {
  if (!isCropping || !cropRect) return;
  isCropping = false;
  $('#uploadedImageContainer').css('cursor', 'default');

  const img = document.getElementById('uploadedImage');
  // Use the actual image's bounding box for accurate alignment
  const imageRect = img.getBoundingClientRect();
  const cropOffset = cropRect.offset();

  // Calculate crop rectangle relative to image's natural size
  const sx = ((cropOffset.left - imageRect.left) / imageRect.width) * img.naturalWidth;
  const sy = ((cropOffset.top - imageRect.top) / imageRect.height) * img.naturalHeight;
  const sw = (cropRect.width() / imageRect.width) * img.naturalWidth;
  const sh = (cropRect.height() / imageRect.height) * img.naturalHeight;

  // Don't crop if width or height is zero or negative
  if (sw <= 0 || sh <= 0) {
    cropRect.remove();
    cropRect = null;
    return;
  }

  // Save current image to history for undo functionality
  const historyImg = new Image();
  historyImg.src = img.src;
  historyImg.className = "rounded border border-secondary shadow-sm";
  historyImg.style.height = "100px";
  historyImg.style.cursor = "pointer";
  historyImg.title = "Previous version";
  $('#imageHistoryWrapper').removeClass('d-none');
  $('#imageHistory').append(historyImg);

  // Draw the cropped region onto a canvas and update the image
  const canvas = document.createElement('canvas');
  canvas.width = sw;
  canvas.height = sh;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);

  img.src = canvas.toDataURL();
  $('#uploadedImage').removeClass('d-none');
  cropRect.remove();
  cropRect = null;

  $('#workingOverlay').removeClass('d-none');
  logWorkingMessage('Rerunning with cropped image...', 'text-white');
  fetchPresign();
});
// --- End Crop Tool Functionality ---

// --- Begin Undo Tool Functionality ---
// Restore previous image from history when undo is clicked
$('#undoToolBtn').on('click', function () {
  const historyImgs = $('#imageHistory img');
  if (historyImgs.length > 0) {
    const lastImg = historyImgs.last();
    const previousSrc = lastImg.attr('src');
    $('#uploadedImage').attr('src', previousSrc).removeClass('d-none');
    lastImg.remove();
  }
});

// --- End Undo Tool Functionality ---

// --- Begin Image History Selection Functionality ---
// When a history image is clicked, make it the current image and rerun the API flow
$('#imageHistory').on('click', 'img', function () {
  const currentImg = $('#uploadedImage')[0];

  // Save the currently displayed image into history
  const historyImg = new Image();
  historyImg.src = currentImg.src;
  historyImg.className = "rounded border border-secondary shadow-sm";
  historyImg.style.height = "100px";
  historyImg.style.cursor = "pointer";
  historyImg.title = "Previous version";
  $('#imageHistory').append(historyImg);

  // Update to the selected history image
  const newSrc = $(this).attr('src');
  $('#uploadedImage').attr('src', newSrc).removeClass('d-none');

  // Rerun the API processing
  $('#workingOverlay').removeClass('d-none');
  logWorkingMessage('Rerunning with selected image from history...', 'text-white');
  fetchPresign();
});
// --- End Image History Selection Functionality ---

// --- Begin Rerun Tool Functionality ---
// Rerun the backend pipeline with the current image
$('#rerunToolBtn').on('click', function () {
  $('#workingOverlay').removeClass('d-none');
  logWorkingMessage('Rerunning with current image...', 'text-white');
  fetchPresign();
});
// --- End Rerun Tool Functionality ---

/**
 * Looks up metadata for a given work ID (e.g., DOI) and displays details.
 * @param {string} work_id - The identifier for the work to look up.
 */
function lookupDOI(work_id) {
  fetch(`${API_BASE_URL}/work/${encodeURIComponent(work_id)}`)
    .then(response => response.json())
    .then(data => {
      showWorkDetails(data);
    })
    .catch(error => {
      console.error("DOI Lookup error:", error);
      alert(`Error looking up DOI for "${work_id}"`);
    });
}

/**
 * Displays a banner with work/DOI details at the top of the page.
 * @param {Object} workData - Metadata object for the work.
 */
function showWorkDetails(workData) {
  // Remove existing banner if present
  $('#workDetailsBanner').remove();

  const details = workData;
  const banner = $(`
    <div id="workDetailsBanner" class="position-fixed top-0 start-0 w-100 bg-info text-white p-3 border-bottom border-dark" style="z-index: 2000;">
      <div class="d-flex justify-content-between align-items-start">
        <div class="w-100">
          <p class="mb-1"><strong>Artist:</strong> ${details.Artist}</p>
          <p class="mb-1"><strong>DOI:</strong> <a href="${details.DOI}" target="_blank" class="text-white text-decoration-underline">${details.DOI}</a></p>
          <p class="mb-1"><strong>Link:</strong> <a href="${details.Link}" target="_blank" class="text-white text-decoration-underline">${details.Link}</a></p>
          <iframe src="${details.DOI}" style="width: 100%; height: 50vh; border: none;" class="mt-2"></iframe>
        </div>
        <button class="btn btn-sm btn-outline-light ms-3" onclick="$('#workDetailsBanner').remove()">
          <i class="bi bi-x-lg"></i>
        </button>
      </div>
    </div>
  `);
  $('body').append(banner);
}