async function predict() {
    const fileInput = document.getElementById("imageInput");
    const preview = document.getElementById("preview");
    const result = document.getElementById("result");
    const loading = document.getElementById("loading");

    const file = fileInput.files[0];
    if (!file) {
        alert("Please select an image");
        return;
    }

    // Preview image
    preview.innerHTML = `<img src="${URL.createObjectURL(file)}">`;

    result.innerText = "";
    result.className = "";
    loading.style.display = "block";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        loading.style.display = "none";

        // IMPORTANT: use same key returned by backend
        if (data.prediction.includes("Car")) {
            result.classList.add("result-car");
        } else {
            result.classList.add("result-bike");
        }

        result.innerText = `Prediction: ${data.prediction}`;

    } catch (error) {
        loading.style.display = "none";
        result.innerText = "Prediction failed";
        console.error(error);
    }
}
