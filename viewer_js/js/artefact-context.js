$(document).ready(function () {
  $('#uploadTrigger').on('click', function () {
    $('#imageUpload').click();
  });

  $('#imageUpload').on('change', function (event) {
    const file = event.target.files[0];
    if (file) {
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

  // Drag and drop support
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
  let selectedSrc = null;

  $('.example-img').on('click', function () {
    $('.example-img').removeClass('selected-img');
    $(this).addClass('selected-img');
    selectedSrc = $(this).attr('src');
    $('#selectImageBtn').removeClass('d-none');
  });

  $('#selectImageBtn').on('click', function () {
    if (selectedSrc) {
      $('#uploadedImage').attr('src', selectedSrc).removeClass('d-none');
      $('#uploadTrigger').addClass('d-none');
      $('#exampleContainer').remove();
      $('#workingOverlay').removeClass('d-none');
      $('#imageTools').removeClass('d-none');
      fetchPresign();
    }
  });
  $('#debugPanelToggle button').on('click', function () {
    $('#debugPanel').toggle();
  });
});

function fetchPresign() {
  $('#debugStatus').text('Requesting ID...');
  $('#workingLog').append('<div class="text-white">Requesting Session ID...</div>');

  fetch('http://127.0.0.1:8000/presign', {
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

      $('#debugSessionId').text(runId);
      $('#debugStatus').text('Got ID');
      $('#workingLog').append('<div class="text-white">Session ID: ' + runId + '</div>');
      $('#workingLog').append('<div class="text-white">Sending /runs request...</div>');
      $('#debugStatus').text('Posting run info');

      fetch('http://127.0.0.1:8000/runs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          runId: runId,
          s3Key: s3Key
        })
      })
        .then(res => res.json())
        .then(response => {
          $('#workingLog').append('<div class="text-white">Run registered successfully</div>');
          $('#debugStatus').text('Run submitted');
          pollRunStatus(runId);
        })
        .catch(err => {
          console.error('Runs error:', err);
          $('#workingLog').append('<div class="text-danger">Error submitting run</div>');
          $('#debugStatus').text('Run submission failed');
        });
    })
    .catch(err => {
      console.error('Presign error:', err);
      $('#debugStatus').text('Error fetching ID');
      $('#workingLog').append('<div class="text-danger">Error fetching ID</div>');
    });
}

function pollRunStatus(runId) {
  $('#workingLog').append('<div class="text-white">Polling run status...</div>');

  const intervalId = setInterval(() => {
    fetch(`http://127.0.0.1:8000/runs/${runId}`)
      .then(res => res.json())
      .then(data => {
        $('#debugStatus').text(`Status: ${data.status}`);
        $('#workingLog').append(`<div class="text-white">Status: ${data.status}</div>`);

        if (data.status !== 'processing') {
          clearInterval(intervalId);
          $('#workingLog').append('<div class="text-white">Processing complete</div>');

          if (data.status === 'done') {
            const filePath = upload?.url?.split('/').pop();
            $('#workingLog').append('<div class="text-white">Fetching outputs from: ' + filePath + '</div>');

            fetch(`http://127.0.0.1:8000/outputs/${filePath}`)
              .then(res => res.json())
              .then(output => {
                $('#workingLog').append('<div class="text-white">Outputs received</div>');
                display_sentences(output);
                $('#workingOverlay').addClass('d-none');
              })
              .catch(err => {
                console.error('Error fetching outputs:', err);
                $('#workingLog').append('<div class="text-danger">Error fetching outputs</div>');
              });
          }
        }
      })
      .catch(err => {
        console.error('Polling error:', err);
        $('#workingLog').append('<div class="text-danger">Error polling status</div>');
        clearInterval(intervalId);
      });
  }, 1000);
}

function escapeHTML(str) {
  return str.replace(/[&<>'"]/g, tag => (
    {'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'}[tag]
  ));
}

function display_sentences(data) {
  console.log("Displaying sentences:", data);
  $('#sentenceList').empty();

  data.forEach((item, index) => {
    const sentenceItem = $(`
      <li class="list-group-item">
        <div class="d-flex">
          <div class="flex-grow-1">
            <p class="mb-1">${escapeHTML(item["sentence"]["English Original"])}
            <button class="btn btn-sm btn-outline-primary" onclick="lookupDOI('${item["sentence"]["Work"]}')" title="Look Up DOI">
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
  let isCropping = false;
  let cropStartX = 0;
  let cropStartY = 0;
  let cropRect = null;

  $('#cropToolBtn').on('click', function () {
    isCropping = true;
    $('#uploadedImageContainer').css('cursor', 'crosshair');
  });

  $('#uploadedImageContainer').on('mousedown', function (e) {
    if (!isCropping) return;
    const rect = this.getBoundingClientRect();
    cropStartX = e.clientX - rect.left;
    cropStartY = e.clientY - rect.top;

    if (cropRect) {
      cropRect.remove();
    }

    cropRect = $('<div>').addClass('position-absolute border border-warning').css({
      left: cropStartX,
      top: cropStartY,
      width: 0,
      height: 0,
      zIndex: 10,
      pointerEvents: 'none'
    }).appendTo('#uploadedImageContainer');
  });

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

  $('#uploadedImageContainer').on('mouseup', function (e) {
    if (!isCropping || !cropRect) return;
    isCropping = false;
    $('#uploadedImageContainer').css('cursor', 'default');

    const img = document.getElementById('uploadedImage');
    // Use the actual image's bounding box for accurate alignment
    const imageRect = img.getBoundingClientRect();
    const cropOffset = cropRect.offset();

    // Calculate crop rectangle relative to image natural size
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

    // Save current image to history
    const historyImg = new Image();
    historyImg.src = img.src;
    historyImg.className = "img-thumbnail";
    historyImg.style.height = "100px";
    $('#imageHistory').append(historyImg);

    const canvas = document.createElement('canvas');
    canvas.width = sw;
    canvas.height = sh;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);

    img.src = canvas.toDataURL();
    $('#uploadedImage').removeClass('d-none');
    cropRect.remove();
    cropRect = null;
  });
  // --- End Crop Tool Functionality ---

  // --- Begin Undo Tool Functionality ---
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

  // --- Begin Rerun Tool Functionality ---
  $('#rerunToolBtn').on('click', function () {
    $('#workingOverlay').removeClass('d-none');
    $('#workingLog').append('<div class="text-white">Rerunning with current image...</div>');
    fetchPresign();
  });
  // --- End Rerun Tool Functionality ---

function lookupDOI(work_id) {
  console.log("Looking up DOI for work:", work_id);

  fetch(`http://127.0.0.1:8000/work/${encodeURIComponent(work_id)}`)
    .then(response => response.json())
    .then(data => {
      console.log("DOI Lookup result:", data);
      showWorkDetails(data);
    })
    .catch(error => {
      console.error("DOI Lookup error:", error);
      alert(`Error looking up DOI for "${work_id}"`);
    });
}

function showWorkDetails(workData) {
  // const workId = Object.keys(workData)[0];
  const details = workData;

  // Remove existing banner if present
  $('#workDetailsBanner').remove();

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