document.addEventListener('DOMContentLoaded', function () {
  let dropArea = document.getElementById('drop-area');
  let fileInput = document.getElementById('fileElem');

  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropArea.addEventListener(eventName, preventDefaults, false);
    document.body.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  dropArea.addEventListener('dragover', () => {
    dropArea.classList.add('highlight');
  }, false);

  dropArea.addEventListener('dragleave', () => {
    dropArea.classList.remove('highlight');
  }, false);

  dropArea.addEventListener('drop', (e) => {
    let dt = e.dataTransfer;
    let files = dt.files;
    if (files.length > 0) {
      fileInput.files = files;
      readAndPreview(files[0]);
    }
  });

  dropArea.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      readAndPreview(fileInput.files[0]);
    }
  });

  function readAndPreview(file) {
    let reader = new FileReader();
    reader.onload = function(e) {
      let img = document.createElement('img');
      img.src = e.target.result;
      img.style.maxWidth = '200px';
      img.style.maxHeight = '200px';
      let prev = dropArea.querySelector('img');
      if (prev) dropArea.removeChild(prev);
      dropArea.appendChild(img);
    };
    reader.readAsDataURL(file);
  }
});
