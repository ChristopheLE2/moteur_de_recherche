let progressBar;
let width = 0;
let interval;


/* début du code réadapté de https://github.com/habibmhamadi/multi-select-tag permettant de sélectionnner plusieurs options dans un dropdown*/
/*********************************************************************************************************************************************/
function MultiSelectTag(el, customs = { shadow: false, rounded: true }) {
    // Initialize variables
    var element = null,
        options = null,
        customSelectContainer = null,
        wrapper = null,
        btnContainer = null,
        body = null,
        inputContainer = null,
        inputBody = null,
        input = null,
        button = null,
        drawer = null,
        ul = null;

    // Customize tag colors
    var tagColor = customs.tagColor || {};
    tagColor.textColor = "#FC3500";
    tagColor.borderColor = "#F78205";
    tagColor.bgColor = "#FFD09D";

    // Initialize DOM Parser
    var domParser = new DOMParser();

    // Initialize
    init();

    function init() {
        // DOM element initialization
        element = document.getElementById(el);
        createElements();
        initOptions();
        enableItemSelection();
        setValues(false);

        // Event listeners
        button.addEventListener('click', () => {
            if (drawer.classList.contains('hidden')) {
                initOptions();
                enableItemSelection();
                drawer.classList.remove('hidden');
                input.focus();
            } else {
                drawer.classList.add('hidden');
            }
        });

        input.addEventListener('keyup', (e) => {
            initOptions(e.target.value);
            enableItemSelection();
        });

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Backspace' && !e.target.value && inputContainer.childElementCount > 1) {
                const child = body.children[inputContainer.childElementCount - 2].firstChild;
                const option = options.find((op) => op.value == child.dataset.value);
                option.selected = false;
                removeTag(child.dataset.value);
                setValues();
            }
        });

        window.addEventListener('click', (e) => {
            if (!customSelectContainer.contains(e.target)) {
                if ((e.target.nodeName !== "LI") && (e.target.getAttribute("class") !== "input_checkbox")) {
                    // hide the list option only if we click outside of it
                    drawer.classList.add('hidden');
                } else {
                    // enable again the click on the list options
                    enableItemSelection();
                }
            }
        });
    }

    function createElements() {
        // Create custom elements
        options = getOptions();
        element.classList.add('hidden');

        // .multi-select-tag
        customSelectContainer = document.createElement('div');
        customSelectContainer.classList.add('mult-select-tag');

        // .container
        wrapper = document.createElement('div');
        wrapper.classList.add('wrapper');

        // body
        body = document.createElement('div');
        body.classList.add('body');
        if (customs.shadow) {
            body.classList.add('shadow');
        }
        if (customs.rounded) {
            body.classList.add('rounded');
        }

        // .input-container
        inputContainer = document.createElement('div');
        inputContainer.classList.add('input-container');

        // input
        input = document.createElement('input');
        input.classList.add('input');
        input.placeholder = `${customs.placeholder || 'Search...'}`;

        inputBody = document.createElement('inputBody');
        inputBody.classList.add('input-body');
        inputBody.append(input);

        body.append(inputContainer);

        // .btn-container
        btnContainer = document.createElement('div');
        btnContainer.classList.add('btn-container');

        // button
        button = document.createElement('button');
        button.type = 'button';
        btnContainer.append(button);

        const icon = domParser.parseFromString(
            `<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="18 15 12 21 6 15"></polyline>
            </svg>`, 'image/svg+xml').documentElement;

        button.append(icon);

        body.append(btnContainer);
        wrapper.append(body);

        drawer = document.createElement('div');
        drawer.classList.add(...['drawer', 'hidden']);
        if (customs.shadow) {
            drawer.classList.add('shadow');
        }
        if (customs.rounded) {
            drawer.classList.add('rounded');
        }
        drawer.append(inputBody);
        ul = document.createElement('ul');

        drawer.appendChild(ul);

        customSelectContainer.appendChild(wrapper);
        customSelectContainer.appendChild(drawer);

        // Place TailwindTagSelection after the element
        if (element.nextSibling) {
            element.parentNode.insertBefore(customSelectContainer, element.nextSibling);
        } else {
            element.parentNode.appendChild(customSelectContainer);
        }
    }

    function createElementInSelectList(option, val, selected = false) {
        // Create a <li> elmt in the drop-down list,
        // selected parameters tells if the checkbox need to be selected and the bg color changed
        const li = document.createElement('li');
        li.innerHTML = "<input type='checkbox' style='margin:0 0.5em 0 0' class='input_checkbox'>"; // add the checkbox at the left of the <li>
        li.innerHTML += option.label;
        li.dataset.value = option.value;
        const checkbox = li.firstChild;
        checkbox.dataset.value = option.value;

        // For search
        if (val && option.label.toLowerCase().startsWith(val.toLowerCase())) {
            ul.appendChild(li);
        } else if (!val) {
            ul.appendChild(li);
        }

        // Change bg color and checking the checkbox
        if (selected) {
            li.style.backgroundColor = tagColor.bgColor;
            checkbox.checked = true;
        }
    }

    function initOptions(val = null) {
        ul.innerHTML = '';
        for (var option of options) {
            // if option already selected
            if (option.selected) {
                !isTagSelected(option.value) && createTag(option);

                // We create a option in the list, but with different color
                createElementInSelectList(option, val, true);
            } else {
                createElementInSelectList(option, val);
            }
        }
    }

    function createTag(option) {
        // Create and show selected item as tag
        const itemDiv = document.createElement('div');
        itemDiv.classList.add('item-container');
        itemDiv.style.color = tagColor.textColor || '#2c7a7b';
        itemDiv.style.borderColor = tagColor.borderColor || '#81e6d9';
        itemDiv.style.background = tagColor.bgColor || '#e6fffa';
        const itemLabel = document.createElement('div');
        itemLabel.classList.add('item-label');
        itemLabel.style.color = tagColor.textColor || '#2c7a7b';
        itemLabel.innerHTML = option.label;
        itemLabel.dataset.value = option.value;
        const itemClose = domParser.parseFromString(
            `<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="item-close-svg">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>`, 'image/svg+xml').documentElement;

        itemClose.addEventListener('click', (e) => {
            const unselectOption = options.find((op) => op.value == option.value);
            unselectOption.selected = false;
            removeTag(option.value);
            initOptions();
            setValues();
        });

        itemDiv.appendChild(itemLabel);
        itemDiv.appendChild(itemClose);
        inputContainer.append(itemDiv);
    }

    function enableItemSelection() {
        // Add click listener to the list items
        for (var li of ul.children) {
            li.addEventListener('click', (e) => {
                if (options.find((o) => o.value == e.target.dataset.value).selected === false) {
                    // if the option is not selected, we select it
                    options.find((o) => o.value == e.target.dataset.value).selected = true;
                    input.value = null;
                    initOptions();
                    setValues();
                    //input.focus() // brings up the list to the input
                } else {
                    // if it's already selected, we deselect it
                    options.find((o) => o.value == e.target.dataset.value).selected = false;
                    input.value = null;
                    initOptions();
                    setValues();
                    //input.focus() // brings up the list on the input
                    removeTag(e.target.dataset.value);
                }
            });
        }
    }

    function isTagSelected(val) {
        // If the item is already selected
        for (var child of inputContainer.children) {
            if (!child.classList.contains('input-body') && child.firstChild.dataset.value == val) {
                return true;
            }
        }
        return false;
    }

    function removeTag(val) {
        // Remove selected item
        for (var child of inputContainer.children) {
            if (!child.classList.contains('input-body') && child.firstChild.dataset.value == val) {
                inputContainer.removeChild(child);
            }
        }
    }

    function setValues(fireEvent = true) {
        // Update element final values
        selected_values = [];
        for (var i = 0; i < options.length; i++) {
            element.options[i].selected = options[i].selected;
            if (options[i].selected) {
                selected_values.push({ label: options[i].label, value: options[i].value });
            }
        }
        if (fireEvent && customs.hasOwnProperty('onChange')) {
            customs.onChange(selected_values);
        }
    }

    function getOptions() {
        // Map element options
        return [...element.options].map((op) => {
            return {
                value: op.value,
                label: op.label,
                selected: op.selected,
            };
        });
    }
}
/*********************************************************************************************************************************************/
/*fin du code réadapté*/




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