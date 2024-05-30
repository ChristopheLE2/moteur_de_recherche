let progressBar;
    let width = 0;
    let interval;

    function afficher()
    {
    alert("Vous venez de cliquer sur le bouton!")
    }

    function notImage()
    {
        alert("Ce fichier n'est pas une image")
    }

    function allowDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.add('dragover');
    }

    function drag(event) {
        event.dataTransfer.setData("text", event.target.id);
    }

    function drop(event) {
        event.preventDefault();
        //afficher();
        var files = event.dataTransfer.files;
        var inImageDisplay = document.getElementById('input_image_display');

        if (files.length > 0) {
            var file = files[0];
            if (file.type.startsWith('image/')) {
                var reader = new FileReader();
                reader.onload = function (e) {
                    inImageDisplay.innerHTML = '<img src="' + e.target.result + '" alt="Uploaded Image" />';
                }
                reader.readAsDataURL(file);
            } else {
                notImage();
            }
        }
    }

    function selectInput() {
        const fileInput = document.getElementById('fileInput');
        const file = fileInput.files[0];
        const imagePreviewContainer = document.getElementById('input_image_display');
        
        if(file.type.match('image.*')){
            const reader = new FileReader();
            
            reader.addEventListener('load', function (event) {
            const imageUrl = event.target.result;
            const image = new Image();
            
            image.addEventListener('load', function() {
                imagePreviewContainer.innerHTML = ''; // Vider le conteneur au cas où il y aurait déjà des images.
                imagePreviewContainer.appendChild(image);
            });
            
            image.src = imageUrl;
            image.style.width = '200px'; // Indiquez les dimensions souhaitées ici.
            image.style.height = 'auto'; // Vous pouvez également utiliser "px" si vous voulez spécifier une hauteur.
            });
            
            reader.readAsDataURL(file);
        }
        else{
            notImage();
        }
    }

    function updateProgressBar() {
        if (width >= 100) {
            clearInterval(interval);
        } else {
            width++;
            progressBar.style.width = width + '%';
            progressBar.textContent = width + '%';
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        progressBar = document.getElementById('progress-bar');
        interval = setInterval(updateProgressBar, 100);
    });