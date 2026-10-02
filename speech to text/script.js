const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const output = document.getElementById("output");
const status = document.getElementById("status");

const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;

if (!SpeechRecognition) {

    status.innerHTML =
        "Speech recognition is not supported.";

} else {

    const recognition = new SpeechRecognition();

    recognition.continuous = true;

    recognition.interimResults = true;

    recognition.lang = "en-US";

    recognition.onstart = function () {

        status.innerHTML = "🎤 Listening...";

    };

    recognition.onresult = function (event) {

        let text = "";

        for (
            let i = event.resultIndex;
            i < event.results.length;
            i++
        ) {

            text +=
                event.results[i][0].transcript;
        }

        output.value = text;

    };

    recognition.onerror = function (event) {

        status.innerHTML =
            "Error: " + event.error;

    };

    recognition.onend = function () {

        status.innerHTML =
            "Speech recognition stopped.";

    };

    startBtn.onclick = function () {

        recognition.start();

    };

    stopBtn.onclick = function () {

        recognition.stop();

    };

}