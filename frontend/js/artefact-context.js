// ==========================
// == GLOBAL CONFIGURATION ==
// ==========================
const API_BASE_URL = 'http://127.0.0.1:8000';

// Variables to store session/run state
let runId;
let imageKey;
let upload;

// --- Grid overlay state ---
let viewGridEnabled = false;

const GRID_ROWS = 7;   // ViT-B/32 → 7×7 patch grid
const GRID_COLS = 7;   // keep rows == cols
const CELL_SIM_K = 10;

// --- Available models list ---
let availableModels = [];
let selectedModel = '';
let creatorsMap = {};

let selectedCreators = [];


// --- Cell highlight state ---
let cellHighlightTimeout = null;

function updateCreatorTags() {
  const tagContainer = $('#creatorTags');
  tagContainer.empty();
  selectedCreators.forEach(name => {
    const tag = $('<span>')
      .addClass('badge bg-primary px-3 py-2 d-flex align-items-center')
      .html(`${name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} <i class="bi bi-x ms-2" style="cursor:pointer;"></i>`);
    tag.find('i').on('click', function () {
      selectedCreators = selectedCreators.filter(c => c !== name);
      updateCreatorTags();
    });
    tagContainer.append(tag);
  });
}

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

// Updates the display of selected creators in the #selectedCreatorsWrapper.
function updateSelectedCreatorsDisplay() {
  $('#selectedCreatorsWrapper').removeClass('d-none');
  const tagContainer = $('#selectedCreatorTags');
  tagContainer.empty();
  selectedCreators.forEach(name => {
    const tag = $('<span>')
      .addClass('badge bg-primary px-3 py-2 d-flex align-items-center')
      .html(`${name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} <i class="bi bi-x ms-2" style="cursor:pointer;"></i>`);
    tag.find('i').on('click', function () {
      selectedCreators = selectedCreators.filter(c => c !== name);
      updateSelectedCreatorsDisplay();
    });
    tagContainer.append(tag);
  });
}

// Main script entry point: sets up event handlers on document ready
$(document).ready(function () {
  // Add click handler for Artefact Viewer logo/text to refresh the app
  $('.navbar-brand').on('click', function(e) {
    e.preventDefault();
    
    // Clear all state variables
    runId = null;
    imageKey = null;
    upload = null;
    viewGridEnabled = false;
    selectedTopics = [];
    selectedCreators = [];
    selectedModel = '';
    
    // Reset UI elements
    $('#uploadedImage').addClass('d-none').attr('src', '');
    $('#uploadTrigger').removeClass('d-none');
    $('#imageTools').addClass('d-none');
    $('#workingOverlay').addClass('d-none');
    $('#workDetailsBanner').remove();
    $('#gridOverlay').hide().html('');
    $('#gridHighlightOverlay').hide();
    
    // Hide panels
    $('.col-md-3').addClass('d-none');
    $('#sentenceList').empty();
    $('#imageHistoryWrapper').addClass('d-none');
    $('#imageHistory').empty();
    $('#selectedTopicsWrapper').addClass('d-none');
    $('#selectedCreatorsWrapper').addClass('d-none');
    
    // Reset topic selections
    $('#topicTags button').removeClass('active btn-primary').addClass('btn-outline-primary');
    $('#selectedTopicTags').empty();
    
    // Reset creator selections  
    $('#creatorTags').empty();
    $('#selectedCreatorTags').empty();
    $('#creatorSearch').val('');
    $('#creatorSearchResults').empty();
    $('#creatorPanelSearch').val('');
    $('#creatorPanelResults').empty();
    
    // Reset model selection to first available
    if (availableModels.length > 0) {
      selectedModel = availableModels[0];
      $('#modelDropdown').text('Model: ' + selectedModel);
      $('#modelDropdownMenu a').removeClass('active');
      $('#modelDropdownMenu a').first().addClass('active');
    }
    
    // Reset debug panel
    $('#debugStatus').text('Idle');
    $('#debugSessionId').text('N/A');
    $('#workingLog').empty();
    
    // Recreate the upload card if it was removed
    if ($('.card:has(#uploadTrigger)').length === 0 && $('#exampleContainer').length === 0) {
      const uploadCard = $(`
        <div class="card h-100 text-center d-flex align-items-center justify-content-center" style="cursor: pointer; background-color: rgba(255,255,255,0.1);">
          <div class="card-body">
            <p class="mb-2">Drop an image here or click to upload</p>
            <button class="btn btn-primary" id="uploadTrigger">
              <i class="bi bi-upload"></i> Upload Image
            </button>
          </div>
        </div>
      `);
      $('#uploadedImageContainer').prepend(uploadCard);
    }
    
    // Show example container if it was hidden
    if ($('#exampleContainer').length === 0) {
      location.reload(); // Simplest way to restore the example images section
    }
  });

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

  // --- Load creators list from /creators ---
  fetch(`${API_BASE_URL}/creators`)
    .then(response => response.json())
    .then(data => {
      creatorsMap = data;
      console.log("Available creators:", creatorsMap);
    })
    .catch(error => {
      console.error('Error loading creators:', error);
    });

  // --- Creator search logic ---
  $('#creatorSearch').on('input', function () {
    const query = $(this).val().toLowerCase();
    const resultsContainer = $('#creatorSearchResults');
    resultsContainer.empty();
    if (query.length > 0) {
      const matches = Object.keys(creatorsMap).filter(name =>
        name.toLowerCase().includes(query)
      );
      matches.forEach((name, index) => {
        const item = $('<button>')
          .addClass('list-group-item list-group-item-action')
          .text(name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()))
          .on('click', function () {
            if (!selectedCreators.includes(name)) {
              selectedCreators.push(name);
              updateCreatorTags();
            }
            $('#creatorSearch').val('');
            resultsContainer.empty();
          });
        if (index === 0) {
          item.addClass('active');
        }
        resultsContainer.append(item);
      });
    }
  });

  // Add enter-to-select first match
  $('#creatorSearch').on('keydown', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      const firstItem = $('#creatorSearchResults button').first();
      if (firstItem.length) {
        firstItem.click();
      }
    }
  });

  // --- Creator panel search logic ---
  $('#creatorPanelSearch').on('input', function () {
    const query = $(this).val().toLowerCase();
    const resultsContainer = $('#creatorPanelResults');
    resultsContainer.empty();
    if (query.length > 0) {
      const matches = Object.keys(creatorsMap).filter(name =>
        name.toLowerCase().includes(query)
      );
      matches.forEach((name, index) => {
        const item = $('<button>')
          .addClass('list-group-item list-group-item-action')
          .text(name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()))
          .on('click', function () {
            if (!selectedCreators.includes(name)) {
              selectedCreators.push(name);
              updateSelectedCreatorsDisplay();
            }
            $('#creatorPanelSearch').val('');
            resultsContainer.empty();
          });
        if (index === 0) {
          item.addClass('active');
        }
        resultsContainer.append(item);
      });
    }
  });

  $('#creatorPanelSearch').on('keydown', function (e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      const firstItem = $('#creatorPanelResults button').first();
      if (firstItem.length) {
        firstItem.click();
      }
    }
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
      imageKey = data.imageKey;
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
          $('#debugStatus').text('Got ID');
          logWorkingMessage('Session ID: ' + runId, 'text-white');
          logWorkingMessage('Sending /runs request...', 'text-white');
          $('#debugStatus').text('Posting run info');

          // Show selected topics box using helper
          updateSelectedTopicsDisplay();
          updateSelectedCreatorsDisplay();

          return fetch(`${API_BASE_URL}/runs`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              runId: runId,
              imageKey: imageKey,
              topics: selectedTopics,
              creators: selectedCreators,
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

  data.forEach(item => {
    const li = $(`
      <li class="list-group-item sentence-item mb-1" data-work="${item["work"]}">
        ${escapeHTML(item["english_original"])}
      </li>
    `);
    // click opens work-details
    li.on('click', function () {
      lookupDOI($(this).data('work'));
    });
    $('#sentenceList').append(li);
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
    .then(res => res.json())
    .then(data => {
      data.Work_ID = work_id;          // +ADD
      showWorkDetails(data);           // unchanged call
    })
    .catch(error => {
      console.error("DOI Lookup error:", error);
      alert(`Error looking up DOI for "${work_id}"`);
    });
}

/**
 * Displays work/DOI details in a centred, scrollable modal rectangle with a dimmed backdrop.
 * @param {Object} workData
 */
function showWorkDetails(workData) {
  // ――― Clean-up any prior overlay ―――
  $('#workOverlayBackdrop, #workDetailsModal').remove();

  const d = workData;

  /* ---------- backdrop (click to close) ---------- */
  const backdrop = $('<div id="workOverlayBackdrop" class="position-fixed top-0 start-0 w-100 h-100" ' +
                     'style="background:rgba(0,0,0,0.5); z-index:2000;"></div>');

  /* ---------- centred rectangle ---------- */
  const modal = $(`
    <div id="workDetailsModal"
         class="position-fixed bg-white border border-primary rounded shadow p-4"
         style="top:50%; left:50%; transform:translate(-50%,-50%);
                max-width:90vw; max-height:80vh; overflow:auto; z-index:2001;">

      <!-- close button -->
      <button type="button"
              class="btn btn-sm btn-outline-secondary position-absolute top-0 end-0 m-2"
              id="workDetailsClose">
        <i class="bi bi-x-lg"></i>
      </button>

      <h5 class="mb-2">${d.Work_Title || 'Unknown Title'}</h5>
      <p class="mb-1"><strong>Author:</strong> ${d.Author_Name || 'Unknown Author'}</p>
      <p class="mb-1"><strong>Year:</strong> ${d.Year || 'Unknown'}</p>

      <!-- Image gallery -->
      <div id="galleryWrapper" class="mb-2">
        <div class="fw-bold">Images in this work</div>
        <div id="galleryScroller" class="mt-1"></div>
      </div>

      <p class="mb-1"><strong>DOI:</strong>
        <a href="${d.DOI}" target="_blank" class="text-primary text-decoration-underline">${d.DOI}</a>
      </p>
      <p class="mb-1"><strong>Link:</strong>
        <a href="${d.Link}" target="_blank" class="text-primary text-decoration-underline">${d.Link}</a>
      </p>

      <!-- BibTeX always visible -->
      <div class="position-relative mt-3">
        <span class="fw-bold">BibTeX Citation</span>
        <button class="btn btn-sm btn-outline-secondary position-absolute top-0 end-0"
                onclick="copyBibTeX()" title="Copy to clipboard">
          <i class="bi bi-clipboard"></i>
        </button>
        <pre id="bibtexContent"
             class="p-2 mt-1 rounded"
             style="white-space:pre-wrap; word-break:break-word;
                    font-size:.875rem; background:#fdfde7; color:#000; padding-right:3rem;">
${d.BibTeX || 'Citation not available'}
        </pre>
      </div>

      <iframe src="${d.DOI}"
              style="width:100%; height:50vh; border:none;"
              class="mt-3"></iframe>
    </div>
  `);

  // inject into DOM
$('body').append(backdrop, modal);

  /* ---------- gallery fetch ---------- */
  if (d.Work_ID) {
    fetch(`${API_BASE_URL}/images/${d.Work_ID}`)
      .then(r => r.json())
      .then(urls => {
        if (!urls.length) { $('#galleryWrapper').hide(); return; }
        const scroller = $('#galleryScroller');
        urls.forEach(u => $('<img>')
          .attr('src', u)
          .attr('crossorigin', 'anonymous')       // ensure CORS safe for canvas
          .addClass('img-thumbnail')
          .css({ height: '120px', cursor: 'pointer' })
          .on('click', () => loadImageAndRun(u))
          .appendTo(scroller));
      })
      .catch(console.error);
  }

  /* ---------- close handlers ---------- */
  backdrop.on('click', () => { backdrop.remove(); modal.remove(); });
  modal.on('click', '#workDetailsClose', () => { backdrop.remove(); modal.remove(); });
}

// Add this helper function for copying BibTeX
function copyBibTeX() {
  const bibtexText = document.getElementById('bibtexContent').textContent.trim();
  
  // Create a temporary textarea to copy from
  const tempTextarea = document.createElement('textarea');
  tempTextarea.value = bibtexText;
  tempTextarea.style.position = 'fixed';
  tempTextarea.style.opacity = '0';
  document.body.appendChild(tempTextarea);
  
  // Select and copy the text
  tempTextarea.select();
  document.execCommand('copy');
  document.body.removeChild(tempTextarea);
  
  // Visual feedback - change icon temporarily
  const copyBtn = event.target.closest('button');
  const icon = copyBtn.querySelector('i');
  icon.classList.remove('bi-clipboard');
  icon.classList.add('bi-clipboard-check');
  
  // Change button text temporarily
  copyBtn.setAttribute('title', 'Copied!');
  
  // Reset after 2 seconds
  setTimeout(() => {
    icon.classList.remove('bi-clipboard-check');
    icon.classList.add('bi-clipboard');
    copyBtn.setAttribute('title', 'Copy to clipboard');
  }, 2000);
}

/**
 * Positions the #gridOverlay to exactly cover the visible image area.
 */
function positionGridOverlayToImage() {
  const container = document.getElementById('uploadedImageContainer');
  const img = document.getElementById('uploadedImage');
  const overlay = document.getElementById('gridOverlay');
  if (!container || !img || !overlay) return;
  if (img.classList.contains('d-none') || !img.src) return;

  const containerRect = container.getBoundingClientRect();
  const imageRect = img.getBoundingClientRect();
  // Compute image rect relative to the container
  const left = imageRect.left - containerRect.left;
  const top = imageRect.top - containerRect.top;

  overlay.style.left = `${left}px`;
  overlay.style.top = `${top}px`;
  overlay.style.width = `${imageRect.width}px`;
  overlay.style.height = `${imageRect.height}px`;
}

/**
 * Draws a 7×7 grid (i.e., 8 vertical + 8 horizontal lines) inside #gridOverlay.
 */
function drawGridOverlay() {
  const overlay = document.getElementById('gridOverlay');
  const img = document.getElementById('uploadedImage');
  if (!overlay || !img || img.classList.contains('d-none') || !img.src) return;

  positionGridOverlayToImage();

  // Clear previous lines
  overlay.innerHTML = '';

  const cols = GRID_COLS;   // 7
  const rows = GRID_ROWS;   // 7

  // Helper to create line
  const makeLine = (styleObj) => {
    const line = document.createElement('div');
    line.style.position = 'absolute';
    line.style.background = 'rgba(255,255,255,0.6)';
    // hairline-ish width
    line.style.boxShadow = '0 0 0 1px rgba(0,0,0,0.1) inset';
    Object.assign(line.style, styleObj);
    overlay.appendChild(line);
  };

  // Vertical lines (9)
  for (let i = 0; i <= cols; i++) {
    const xPct = (i / cols) * 100;
    makeLine({
      top: '0',
      bottom: '0',
      width: '1px',
      left: `calc(${xPct}% - 0.5px)`,
    });
  }

  // Horizontal lines (9)
  for (let j = 0; j <= rows; j++) {
    const yPct = (j / rows) * 100;
    makeLine({
      left: '0',
      right: '0',
      height: '1px',
      top: `calc(${yPct}% - 0.5px)`,
    });
  }
}

/**
 * Shows/hides and (re)draws the grid depending on toggle state.
 */
function updateGridVisibility() {
  const overlay = document.getElementById('gridOverlay');
  if (!overlay) return;
  if (viewGridEnabled) {
    overlay.style.display = 'block';
    drawGridOverlay();
  } else {
    overlay.style.display = 'none';
    overlay.innerHTML = '';
  }
}

// Ensure the toggle reflects current state when modal opens
$('#settingsModal').on('shown.bs.modal', function () {
  $('#toggleViewGrid').prop('checked', viewGridEnabled);
});

// Toggle handler
$(document).on('change', '#toggleViewGrid', function () {
  viewGridEnabled = $(this).is(':checked');
  updateGridVisibility();
});

$(window).on('resize', function () {
  if (viewGridEnabled) {
    drawGridOverlay();
  }
});
// Redraw the grid whenever the image finishes loading / changes
$('#uploadedImage').on('load', function () {
    if (viewGridEnabled) drawGridOverlay();
    updateGridVisibility(); // positions + draws
  const hi = document.getElementById('gridHighlightOverlay');
  if (hi) { hi.style.display = 'none'; }
});


function getGridCellFromClick(event) {
  const img = document.getElementById('uploadedImage');
  if (!img || img.classList.contains('d-none') || !img.src) return null;

  const rect = img.getBoundingClientRect();
  const x = event.clientX;
  const y = event.clientY;

  if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
    return null; // clicked outside visible image bounds
  }

  const dx = (x - rect.left) / rect.width;  // 0..1
  const dy = (y - rect.top) / rect.height;  // 0..1

  let col = Math.floor(dx * GRID_COLS);
  let row = Math.floor(dy * GRID_ROWS);

  // Clamp just in case of boundary rounding
  col = Math.max(0, Math.min(GRID_COLS - 1, col));
  row = Math.max(0, Math.min(GRID_ROWS - 1, row));

  return { row, col };
}

$('#uploadedImageContainer').on('click', function (e) {
  // Ignore if cropping in progress
  if (typeof isCropping !== 'undefined' && isCropping) return;

  if (!runId) {
    logWorkingMessage('No run active yet. Upload/select an image first.', 'text-danger');
    return;
  }

  const cell = getGridCellFromClick(e);
  if (!cell) return;

  const { row, col } = cell;

  // NEW: spatial feedback
  showCellHighlight(row, col);

  logWorkingMessage(`Cell click → row ${row}, col ${col}. Requesting /cell-sim...`, 'text-white');

const params = new URLSearchParams({
  runId: runId,
  row: String(row),
  col: String(col),
  K: String(CELL_SIM_K)
});

fetch(`${API_BASE_URL}/cell-sim?${params.toString()}`)
  .then(res => res.json())
  .then(data => {
    logWorkingMessage('Cell similarities received.', 'text-white');
    display_sentences(data);
  })
  .catch(err => {
    console.error('cell-sim error:', err);
    logWorkingMessage('Error fetching cell similarities.', 'text-danger');
  });
});


/**
 * Briefly highlight a specific grid cell on the visible image.
 * @param {number} row - 0..GRID_ROWS-1
 * @param {number} col - 0..GRID_COLS-1
 */
function showCellHighlight(row, col) {
  const container = document.getElementById('uploadedImageContainer');
  const img = document.getElementById('uploadedImage');
  const hi = document.getElementById('gridHighlightOverlay');
  if (!container || !img || !hi) return;
  if (img.classList.contains('d-none') || !img.src) return;

  // Position relative to container, aligned to visible image rect.
  const containerRect = container.getBoundingClientRect();
  const imageRect = img.getBoundingClientRect();

  const cellW = imageRect.width / GRID_COLS;
  const cellH = imageRect.height / GRID_ROWS;

  const left = (imageRect.left - containerRect.left) + col * cellW;
  const top  = (imageRect.top  - containerRect.top)  + row * cellH;

  // Style as an outline box with subtle fill, and fade-out transition.
  hi.style.left = `${left}px`;
  hi.style.top = `${top}px`;
  hi.style.width = `${cellW}px`;
  hi.style.height = `${cellH}px`;
  hi.style.border = '2px solid rgba(255, 255, 0, 0.9)';
  hi.style.boxShadow = '0 0 0 1px rgba(0,0,0,0.25) inset';
  hi.style.background = 'rgba(255, 255, 0, 0.10)';
  hi.style.opacity = '1';
  hi.style.transition = 'opacity 200ms ease';
  hi.style.display = 'block';

  // Clear any previous timer, then fade out and hide.
  if (cellHighlightTimeout) clearTimeout(cellHighlightTimeout);
  cellHighlightTimeout = setTimeout(() => {
    hi.style.opacity = '0';
    setTimeout(() => {
      hi.style.display = 'none';
    }, 210);
  }, 600);
}

// ──────────────────────────────────────────────────────────────────────────────
//  NEW helper : use a gallery image as the next run
// ──────────────────────────────────────────────────────────────────────────────
function loadImageAndRun(imgSrc) {
  // close the modal/backdrop if still open
  $('#workOverlayBackdrop, #workDetailsModal').remove();

  // show the chosen artwork in the main image slot
  const $img = $('#uploadedImage')
                 .attr('src', imgSrc)
                 .attr('crossorigin', 'anonymous')   // allow canvas use
                 .removeClass('d-none');

  // hide the upload card / example images just like other entry paths
  $('#uploadTrigger').addClass('d-none');
  $('.card:has(#uploadTrigger), #exampleContainer').remove();

  // UI bits the normal flow expects
  $('#workingOverlay').removeClass('d-none');
  $('#imageTools').removeClass('d-none');

  // make sure we fetch a presign only after the image data is ready
  $img.one('load', () => fetchPresign());
}