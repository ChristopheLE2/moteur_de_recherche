let progressBar;
let width = 0;
let interval;

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

function updateProgressBar_old() {
    if (width >= 100) {
        clearInterval(interval);
    } else {
        width++;
        progressBar.style.width = width + '%';
        progressBar.textContent = width + '%';
    }
}


function updateProgressBar() {
    fetch('/progress')
        .then(response => response.json())
        .then(data => {
            const progress = data.progress;
            const progressBar = document.getElementById('progress-bar');
            progressBar.style.width = progress + '%';
            progressBar.textContent = progress + '%';

            if (progress < 100) {
                setTimeout(updateProgressBar, 500);
            }
        });
}





function loadImages() {
    fetch('/get-images')
        .then(response => response.json())
        .then(data => {
            const imageContainer = document.getElementById('results');
            imageContainer.innerHTML = ''; // Clear previous images
            data.forEach(image => {
                const imgElement = document.createElement('img');
                imgElement.src = `/static/${image}`;
                imageContainer.appendChild(imgElement);
            });

            const RP_Container = document.getElementById('RP_pic');
            RP_Container.innerHTML = ''; // Clear previous images
            const imgElement = document.createElement('img');
            imgElement.src = `/static/new_plot.png`;
            imgElement.id = "RP_graph"
            RP_Container.appendChild(imgElement);
        })
        .catch(error => console.error('Error fetching images:', error));

}



document.addEventListener('DOMContentLoaded', () => {
    progressBar = document.getElementById('progress-bar');
    interval = setInterval(updateProgressBar, 100);
});