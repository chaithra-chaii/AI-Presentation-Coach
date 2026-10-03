/* =====================================================
   AI PRESENTATION COACH
   MEMBER 1
   SPEECH + SPEECH-TO-TEXT

   FEATURES

   1. Microphone capture
   2. English speech recognition
   3. Live transcript
   4. Timestamps
   5. Speaking duration
   6. Speaking speed / WPM
   7. Pause detection > 2 seconds
   8. Filler word detection
   9. Recording
   10. Recording playback
   11. Timestamp playback
   12. Backend JSON
   13. Stop freezes timer
===================================================== */



/* =====================================================
   1. HTML ELEMENTS
===================================================== */

const startBtn =
    document.getElementById("startBtn");

const pauseBtn =
    document.getElementById("pauseBtn");

const resumeBtn =
    document.getElementById("resumeBtn");

const stopBtn =
    document.getElementById("stopBtn");

const clearBtn =
    document.getElementById("clearBtn");

const languageSelect =
    document.getElementById("languageSelect");

const transcriptBox =
    document.getElementById("transcript");

const fillerList =
    document.getElementById("fillerList");

const pauseList =
    document.getElementById("pauseList");

const speakingTime =
    document.getElementById("speakingTime");

const wpmDisplay =
    document.getElementById("wpm");

const pauseCountDisplay =
    document.getElementById("pauseCount");

const fillerCountDisplay =
    document.getElementById("fillerCount");

const audioPlayer =
    document.getElementById("audioPlayer");

const recordingInfo =
    document.getElementById("recordingInfo");

const jsonOutput =
    document.getElementById("jsonOutput");

const statusDisplay =
    document.getElementById("status");



/* =====================================================
   2. SPEECH RECOGNITION SUPPORT
===================================================== */

const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;



/* =====================================================
   3. VARIABLES
===================================================== */


/* Speech recognition */

let recognition = null;


/* Microphone */

let microphoneStream = null;


/* Audio recorder */

let mediaRecorder = null;

let audioChunks = [];

let audioURL = null;



/* =====================================================
   PRESENTATION TIMER VARIABLES
===================================================== */

let presentationStartTime = null;


/*
   Total manually paused time.
*/
let totalPausedTime = 0;


/*
   Time at which manual pause started.
*/
let manualPauseStart = null;


/*
   TRUE while presentation is running.
*/
let presentationActive = false;


/*
   Final time after Stop.
*/
let finalSpeakingTime = 0;


/*
   Timer ID.
*/
let timerInterval = null;



/* =====================================================
   TRANSCRIPT VARIABLES
===================================================== */

let transcriptSegments = [];

let completeTranscript = "";

let temporaryTranscript = "";



/* =====================================================
   PAUSE VARIABLES
===================================================== */

let detectedPauses = [];

let lastSpeechTime = null;

let pauseStarted = null;


/*
   Pause threshold = 2 seconds.
*/
const PAUSE_THRESHOLD = 2000;



/* =====================================================
   FILLER WORDS
===================================================== */

let fillerWords = {

    "um": 0,

    "uh": 0,

    "er": 0,

    "ah": 0,

    "actually": 0,

    "basically": 0,

    "like": 0,

    "you know": 0,

    "i mean": 0,

    "sort of": 0,

    "kind of": 0

};



const FILLER_WORD_LIST = [

    "um",

    "uh",

    "er",

    "ah",

    "actually",

    "basically",

    "like",

    "you know",

    "i mean",

    "sort of",

    "kind of"

];



/* =====================================================
   4. FORMAT TIME
===================================================== */

function formatTime(seconds) {

    seconds =
        Math.max(
            0,
            Math.floor(seconds)
        );


    const minutes =
        Math.floor(seconds / 60)
            .toString()
            .padStart(2, "0");


    const secs =
        (seconds % 60)
            .toString()
            .padStart(2, "0");


    return `${minutes}:${secs}`;

}



/* =====================================================
   5. GET CURRENT PRESENTATION TIME
===================================================== */

function getPresentationTime() {


    /*
       IMPORTANT:

       When presentation is stopped,
       return the frozen final time.

       This is what fixes the Stop
       button timer problem.
    */

    if (!presentationActive) {

        return finalSpeakingTime;

    }


    if (!presentationStartTime) {

        return 0;

    }


    let currentTime =
        Date.now() -
        presentationStartTime;


    /*
       Remove manual pause time.

       If currently manually paused,
       also remove the current pause.
    */

    if (manualPauseStart !== null) {

        currentTime -=
            totalPausedTime;

        currentTime -=
            Date.now() -
            manualPauseStart;

    }
    else {

        currentTime -=
            totalPausedTime;

    }


    return Math.max(
        0,
        currentTime / 1000
    );

}



/* =====================================================
   6. COUNT WORDS
===================================================== */

function countWords(text) {

    if (!text.trim()) {

        return 0;

    }


    return text
        .trim()
        .split(/\s+/)
        .length;

}



/* =====================================================
   7. CALCULATE WPM
===================================================== */

function calculateWPM() {

    const words =
        countWords(
            completeTranscript
        );


    const time =
        getPresentationTime();


    if (
        words === 0 ||
        time <= 0
    ) {

        return 0;

    }


    const minutes =
        time / 60;


    return Math.round(
        words / minutes
    );

}



/* =====================================================
   8. FILLER WORD DETECTION
===================================================== */

function detectFillerWords(text) {

    const lowerText =
        text.toLowerCase();


    FILLER_WORD_LIST.forEach(
        filler => {


            const escaped =
                filler.replace(
                    /[.*+?^${}()|[\]\\]/g,
                    "\\$&"
                );


            const pattern =
                new RegExp(
                    `\\b${escaped}\\b`,
                    "gi"
                );


            const matches =
                lowerText.match(
                    pattern
                );


            if (matches) {

                fillerWords[filler] +=
                    matches.length;

            }

        }
    );

}



/* =====================================================
   9. TOTAL FILLER WORDS
===================================================== */

function getTotalFillerWords() {

    return Object.values(
        fillerWords
    ).reduce(
        (total, value) =>
            total + value,
        0
    );

}



/* =====================================================
   10. CREATE SPEECH RECOGNITION
===================================================== */

function createSpeechRecognition() {


    if (!SpeechRecognition) {

        alert(
            "Speech recognition is not supported. Please use Google Chrome or Microsoft Edge."
        );

        return null;

    }


    const speech =
        new SpeechRecognition();


    /*
       Continue listening.
    */

    speech.continuous = true;


    /*
       Show partial speech.
    */

    speech.interimResults = true;


    /*
       English only.
    */

    speech.lang =
        languageSelect.value;


    speech.maxAlternatives = 1;



    /* =================================================
       RECOGNITION START
    ================================================= */

    speech.onstart = function() {

        console.log(
            "Speech recognition started."
        );

    };



    /* =================================================
       SPEECH RESULT
    ================================================= */

    speech.onresult =
        function(event) {


            let interimText = "";


            for (
                let i = event.resultIndex;
                i < event.results.length;
                i++
            ) {


                const result =
                    event.results[i];


                /*
                   FINAL SPEECH
                */

                if (result.isFinal) {


                    const text =
                        result[0]
                            .transcript
                            .trim();


                    if (!text) {

                        continue;

                    }


                    /*
                       Timestamp.
                    */

                    const timestamp =
                        getPresentationTime();


                    /*
                       Save transcript segment.
                    */

                    transcriptSegments.push({

                        timestamp:
                            Number(
                                timestamp.toFixed(2)
                            ),

                        text:
                            text

                    });


                    /*
                       Add to complete transcript.
                    */

                    completeTranscript +=
                        " " + text;


                    /*
                       Detect filler words.
                    */

                    detectFillerWords(
                        text
                    );


                    /*
                       Update speech activity.
                    */

                    lastSpeechTime =
                        Date.now();


                    /*
                       If a long pause was
                       detected, register it.
                    */

                    if (
                        pauseStarted !== null
                    ) {

                        registerPause(
                            pauseStarted,
                            Date.now()
                        );

                        pauseStarted = null;

                    }


                    updateInterface();

                }

                else {

                    /*
                       INTERIM SPEECH
                    */

                    interimText +=
                        result[0]
                            .transcript;

                }

            }


            temporaryTranscript =
                interimText;


            displayTranscript();

        };



    /* =================================================
       RECOGNITION ERROR
    ================================================= */

    speech.onerror =
        function(event) {

            console.log(
                "Speech recognition error:",
                event.error
            );


            if (
                event.error ===
                "not-allowed"
            ) {

                alert(
                    "Microphone permission was denied."
                );

            }

        };



    /* =================================================
       RECOGNITION ENDED
    ================================================= */

    speech.onend =
        function() {


            /*
               Restart ONLY if presentation
               is still active.

               This prevents recognition
               from restarting after Stop.
            */

            if (
                presentationActive &&
                manualPauseStart === null
            ) {

                try {

                    speech.start();

                }
                catch(error) {

                    console.log(
                        "Recognition restart:",
                        error
                    );

                }

            }

        };


    return speech;

}



/* =====================================================
   11. START PRESENTATION
===================================================== */

async function startPresentation() {

    try {


        /*
           Ask for microphone permission.
        */

        microphoneStream =
            await navigator
                .mediaDevices
                .getUserMedia({

                    audio: {

                        echoCancellation: true,

                        noiseSuppression: true,

                        autoGainControl: true

                    }

                });



        /*
           Reset presentation data.
        */

        audioChunks = [];

        transcriptSegments = [];

        completeTranscript = "";

        temporaryTranscript = "";

        detectedPauses = [];

        lastSpeechTime =
            Date.now();

        pauseStarted = null;


        /*
           Reset filler words.
        */

        fillerWords = {

            "um": 0,

            "uh": 0,

            "er": 0,

            "ah": 0,

            "actually": 0,

            "basically": 0,

            "like": 0,

            "you know": 0,

            "i mean": 0,

            "sort of": 0,

            "kind of": 0

        };


        /*
           Reset timer.
        */

        presentationStartTime =
            Date.now();

        totalPausedTime = 0;

        manualPauseStart = null;


        /*
           IMPORTANT:

           Presentation is now active.
        */

        presentationActive = true;


        /*
           Reset frozen time.
        */

        finalSpeakingTime = 0;



        /* =================================================
           MEDIA RECORDER
        ================================================= */

        let recorderOptions = {};


        /*
           Use WebM when supported.
        */

        if (
            MediaRecorder.isTypeSupported(
                "audio/webm;codecs=opus"
            )
        ) {

            recorderOptions = {

                mimeType:
                    "audio/webm;codecs=opus"

            };

        }


        mediaRecorder =
            new MediaRecorder(
                microphoneStream,
                recorderOptions
            );


        mediaRecorder.ondataavailable =
            function(event) {

                if (
                    event.data.size > 0
                ) {

                    audioChunks.push(
                        event.data
                    );

                }

            };


        mediaRecorder.onstop =
            function() {

                createRecording();

            };


        mediaRecorder.start();



        /* =================================================
           SPEECH RECOGNITION
        ================================================= */

        recognition =
            createSpeechRecognition();


        if (recognition) {

            recognition.start();

        }



        /* =================================================
           BUTTONS
        ================================================= */

        startBtn.disabled = true;

        pauseBtn.disabled = false;

        resumeBtn.disabled = true;

        stopBtn.disabled = false;



        /* =================================================
           STATUS
        ================================================= */

        statusDisplay.textContent =
            "● Live Analysis";

        statusDisplay.className =
            "status live";



        /*
           Start automatic pause detection.
        */

        startSilenceDetection();


        /*
           Start timer.
        */

        updateTimer();


        /*
           Update interface immediately.
        */

        updateInterface();

    }
    catch(error) {

        console.error(error);


        /*
           If microphone permission fails,
           reset presentation state.
        */

        presentationActive = false;

        presentationStartTime = null;


        alert(
            "Microphone access is required. Please allow microphone permission and try again."
        );

    }

}



/* =====================================================
   12. MANUAL PAUSE
===================================================== */

function pausePresentation() {


    if (
        !mediaRecorder ||
        mediaRecorder.state !== "recording"
    ) {

        return;

    }


    /*
       Store pause starting time.
    */

    manualPauseStart =
        Date.now();


    mediaRecorder.pause();


    /*
       Stop speech recognition temporarily.
    */

    if (recognition) {

        try {

            recognition.stop();

        }
        catch(error) {

            console.log(error);

        }

    }


    pauseBtn.disabled = true;

    resumeBtn.disabled = false;


    statusDisplay.textContent =
        "● Paused";

    statusDisplay.className =
        "status paused";


    /*
       Update timer so it freezes.
    */

    updateInterface();

}



/* =====================================================
   13. RESUME PRESENTATION
===================================================== */

function resumePresentation() {


    if (
        !mediaRecorder ||
        mediaRecorder.state !== "paused"
    ) {

        return;

    }


    const resumeTime =
        Date.now();


    /*
       Add manual pause duration.
    */

    if (manualPauseStart !== null) {

        totalPausedTime +=
            resumeTime -
            manualPauseStart;

    }


    manualPauseStart = null;


    mediaRecorder.resume();


    /*
       Create new recognition instance.
    */

    recognition =
        createSpeechRecognition();


    if (recognition) {

        try {

            recognition.start();

        }
        catch(error) {

            console.log(error);

        }

    }


    pauseBtn.disabled = false;

    resumeBtn.disabled = true;


    statusDisplay.textContent =
        "● Live Analysis";

    statusDisplay.className =
        "status live";


    updateInterface();

}



/* =====================================================
   14. STOP PRESENTATION
===================================================== */

function stopPresentation() {


    /*
       If there is no active presentation,
       do nothing.
    */

    if (
        !presentationStartTime
    ) {

        return;

    }



    /* =================================================
       STEP 1
       CALCULATE FINAL TIME
    ================================================= */


    /*
       If manually paused when Stop is clicked,
       calculate the final active speaking time
       correctly.
    */

    finalSpeakingTime =
        getPresentationTime();


    /*
       Now mark presentation as inactive.

       From this point onward,
       getPresentationTime() returns
       finalSpeakingTime.
    */

    presentationActive = false;



    /* =================================================
       STEP 2
       STOP TIMER
    ================================================= */

    if (timerInterval !== null) {

        clearTimeout(
            timerInterval
        );

        timerInterval = null;

    }



    /* =================================================
       STEP 3
       FINISH AUTOMATIC PAUSE
    ================================================= */

    if (
        pauseStarted !== null
    ) {

        registerPause(
            pauseStarted,
            Date.now()
        );

        pauseStarted = null;

    }



    /* =================================================
       STEP 4
       STOP SPEECH RECOGNITION
    ================================================= */

    if (recognition) {

        try {

            /*
               Remove onend handler first.

               This is important because
               otherwise recognition could
               restart after Stop.
            */

            recognition.onend = null;

            recognition.stop();

        }
        catch(error) {

            console.log(
                "Recognition stop:",
                error
            );

        }

    }



    /* =================================================
       STEP 5
       STOP AUDIO RECORDING
    ================================================= */

    if (
        mediaRecorder &&
        mediaRecorder.state !== "inactive"
    ) {

        mediaRecorder.stop();

    }



    /* =================================================
       STEP 6
       STOP MICROPHONE
    ================================================= */

    if (microphoneStream) {

        microphoneStream
            .getTracks()
            .forEach(
                track =>
                    track.stop()
            );

    }



    /* =================================================
       STEP 7
       BUTTONS
    ================================================= */

    startBtn.disabled = false;

    pauseBtn.disabled = true;

    resumeBtn.disabled = true;

    stopBtn.disabled = true;



    /* =================================================
       STEP 8
       STATUS
    ================================================= */

    statusDisplay.textContent =
        "● Finished";

    statusDisplay.className =
        "status finished";



    /* =================================================
       STEP 9
       DISPLAY FINAL TIME
    ================================================= */

    speakingTime.textContent =
        formatTime(
            finalSpeakingTime
        );



    /* =================================================
       STEP 10
       UPDATE FINAL DATA
    ================================================= */

    updateInterface();

    updateBackendJSON();


    console.log(
        "Presentation stopped."
    );

    console.log(
        "Final speaking time:",
        finalSpeakingTime
    );

}



/* =====================================================
   15. CREATE RECORDING
===================================================== */

function createRecording() {


    if (
        audioChunks.length === 0
    ) {

        return;

    }


    const mimeType =
        mediaRecorder &&
        mediaRecorder.mimeType
            ? mediaRecorder.mimeType
            : "audio/webm";


    const audioBlob =
        new Blob(
            audioChunks,
            {
                type: mimeType
            }
        );


    /*
       Remove old URL.
    */

    if (audioURL) {

        URL.revokeObjectURL(
            audioURL
        );

    }


    /*
       Create new playback URL.
    */

    audioURL =
        URL.createObjectURL(
            audioBlob
        );


    audioPlayer.src =
        audioURL;


    recordingInfo.textContent =
        "Recording ready. Click a timestamp to jump to that point.";

}



/* =====================================================
   16. AUTOMATIC SILENCE DETECTION
===================================================== */

function startSilenceDetection() {


    if (!microphoneStream) {

        return;

    }


    const AudioContextClass =
        window.AudioContext ||
        window.webkitAudioContext;


    if (!AudioContextClass) {

        console.log(
            "AudioContext is not supported."
        );

        return;

    }


    const audioContext =
        new AudioContextClass();


    const source =
        audioContext
            .createMediaStreamSource(
                microphoneStream
            );


    const analyser =
        audioContext
            .createAnalyser();


    analyser.fftSize = 2048;


    source.connect(
        analyser
    );


    const data =
        new Uint8Array(
            analyser.fftSize
        );


    function checkAudio() {


        /*
           Stop checking after presentation
           has ended.
        */

        if (!presentationActive) {

            try {

                audioContext.close();

            }
            catch(error) {}

            return;

        }


        analyser.getByteTimeDomainData(
            data
        );


        /*
           Calculate microphone volume.
        */

        let sum = 0;


        for (
            let i = 0;
            i < data.length;
            i++
        ) {

            const value =
                (data[i] - 128) /
                128;


            sum +=
                value * value;

        }


        const rms =
            Math.sqrt(
                sum / data.length
            );


        const currentTime =
            Date.now();


        /*
           Speech volume threshold.

           This can be adjusted later
           depending on microphone.
        */

        const SPEECH_THRESHOLD =
            0.025;


        const userIsSpeaking =
            rms >
            SPEECH_THRESHOLD;



        /* =================================================
           USER IS SPEAKING
        ================================================= */

        if (userIsSpeaking) {


            /*
               If a pause was already
               detected, register it.
            */

            if (
                pauseStarted !== null
            ) {

                registerPause(
                    pauseStarted,
                    currentTime
                );

                pauseStarted = null;

            }


            lastSpeechTime =
                currentTime;

        }



        /* =================================================
           USER IS SILENT
        ================================================= */

        else {


            if (
                lastSpeechTime !== null
            ) {


                const silenceDuration =
                    currentTime -
                    lastSpeechTime;


                /*
                   More than 2 seconds?
                */

                if (
                    silenceDuration >
                    PAUSE_THRESHOLD
                ) {


                    if (
                        pauseStarted === null
                    ) {

                        pauseStarted =
                            lastSpeechTime;

                    }

                }

            }

        }


        requestAnimationFrame(
            checkAudio
        );

    }


    checkAudio();

}



/* =====================================================
   17. REGISTER LONG PAUSE
===================================================== */

function registerPause(
    startTime,
    endTime
) {


    const duration =
        endTime -
        startTime;


    /*
       Only count pauses >= 2 seconds.
    */

    if (
        duration <
        PAUSE_THRESHOLD
    ) {

        return;

    }


    if (!presentationStartTime) {

        return;

    }


    /*
       Convert pause time into
       presentation time.
    */

    const startSeconds =
        (
            startTime -
            presentationStartTime -
            totalPausedTime
        ) / 1000;


    const endSeconds =
        (
            endTime -
            presentationStartTime -
            totalPausedTime
        ) / 1000;


    const pause = {

        start:
            Number(
                Math.max(
                    0,
                    startSeconds
                ).toFixed(2)
            ),

        end:
            Number(
                Math.max(
                    0,
                    endSeconds
                ).toFixed(2)
            ),

        duration:
            Number(
                (
                    duration / 1000
                ).toFixed(2)
            )

    };


    detectedPauses.push(
        pause
    );


    displayPauses();

    updateInterface();

}



/* =====================================================
   18. DISPLAY TRANSCRIPT
===================================================== */

function displayTranscript() {


    if (
        transcriptSegments.length === 0 &&
        !temporaryTranscript
    ) {

        transcriptBox.innerHTML = `

            <div class="empty-message">

                Your presentation transcript
                will appear here...

            </div>

        `;

        return;

    }


    let html = "";



    /*
       Final transcript segments.
    */

    transcriptSegments.forEach(
        segment => {


            html += `

                <div class="transcript-line">

                    <button
                        class="timestamp"
                        onclick="jumpToTimestamp(
                            ${segment.timestamp}
                        )">

                        ${formatTime(
                            segment.timestamp
                        )}

                    </button>


                    <div class="transcript-text">

                        ${escapeHTML(
                            segment.text
                        )}

                    </div>

                </div>

            `;

        }
    );



    /*
       Temporary speech.
    */

    if (temporaryTranscript) {

        html += `

            <div class="transcript-line">

                <div class="timestamp">
                    ...
                </div>

                <div class="transcript-text">

                    ${escapeHTML(
                        temporaryTranscript
                    )}

                </div>

            </div>

        `;

    }


    transcriptBox.innerHTML =
        html;


    /*
       Scroll to newest speech.
    */

    transcriptBox.scrollTop =
        transcriptBox.scrollHeight;

}



/* =====================================================
   19. ESCAPE HTML
===================================================== */

function escapeHTML(text) {

    return text

        .replaceAll(
            "&",
            "&amp;"
        )

        .replaceAll(
            "<",
            "&lt;"
        )

        .replaceAll(
            ">",
            "&gt;"
        )

        .replaceAll(
            '"',
            "&quot;"
        )

        .replaceAll(
            "'",
            "&#039;"
        );

}



/* =====================================================
   20. DISPLAY FILLER WORDS
===================================================== */

function displayFillerWords() {


    const entries =
        Object.entries(
            fillerWords
        )
        .filter(
            ([word, count]) =>
                count > 0
        )
        .sort(
            (a, b) =>
                b[1] - a[1]
        );


    if (
        entries.length === 0
    ) {

        fillerList.innerHTML = `

            <div class="empty-message">

                No filler words detected.

            </div>

        `;

        return;

    }


    let html = "";


    entries.forEach(
        ([word, count]) => {

            html += `

                <div class="filler-row">

                    <span>
                        ${escapeHTML(word)}
                    </span>

                    <strong>
                        ${count}
                    </strong>

                </div>

            `;

        }
    );


    fillerList.innerHTML =
        html;

}



/* =====================================================
   21. DISPLAY PAUSES
===================================================== */

function displayPauses() {


    if (
        detectedPauses.length === 0
    ) {

        pauseList.innerHTML = `

            <div class="empty-message">

                No long pauses detected.

            </div>

        `;

        return;

    }


    let html = "";


    detectedPauses.forEach(
        pause => {

            html += `

                <div class="pause-row">

                    <span>

                        ${formatTime(
                            pause.start
                        )}

                        →

                        ${formatTime(
                            pause.end
                        )}

                    </span>


                    <strong>

                        ${pause.duration}s

                    </strong>

                </div>

            `;

        }
    );


    pauseList.innerHTML =
        html;

}



/* =====================================================
   22. UPDATE INTERFACE
===================================================== */

function updateInterface() {


    /*
       Speaking time.
    */

    const time =
        getPresentationTime();


    speakingTime.textContent =
        formatTime(time);


    /*
       WPM.
    */

    const wpm =
        calculateWPM();


    wpmDisplay.textContent =
        wpm;


    /*
       Pause count.
    */

    pauseCountDisplay.textContent =
        detectedPauses.length;


    /*
       Filler count.
    */

    fillerCountDisplay.textContent =
        getTotalFillerWords();


    /*
       Update lists.
    */

    displayFillerWords();

    displayPauses();


    /*
       Backend data.
    */

    updateBackendJSON();

}



/* =====================================================
   23. TIMER
===================================================== */

function updateTimer() {


    /*
       IMPORTANT:

       If Stop was clicked,
       do not start another timer.
    */

    if (!presentationActive) {

        return;

    }


    /*
       Update display.
    */

    updateInterface();


    /*
       Run again after 1 second.
    */

    timerInterval =
        setTimeout(
            updateTimer,
            1000
        );

}



/* =====================================================
   24. TIMESTAMP PLAYBACK
===================================================== */

function jumpToTimestamp(
    seconds
) {


    if (!audioPlayer.src) {

        alert(
            "Recording is not available yet."
        );

        return;

    }


    /*
       Jump to timestamp.
    */

    audioPlayer.currentTime =
        seconds;


    /*
       Start playback.
    */

    audioPlayer.play()
        .catch(
            error =>
                console.log(error)
        );

}



/* =====================================================
   25. CREATE BACKEND DATA
===================================================== */

function createBackendData() {


    return {

        /*
           Transcript
        */

        transcript:
            completeTranscript.trim(),


        /*
           Speaking time
        */

        speaking_time:
            Number(
                getPresentationTime()
                    .toFixed(2)
            ),


        speaking_time_formatted:
            formatTime(
                getPresentationTime()
            ),


        /*
           WPM
        */

        words_per_minute:
            calculateWPM(),


        /*
           Pause information
        */

        pause_count:
            detectedPauses.length,


        pauses:
            detectedPauses,


        /*
           Filler words
        */

        filler_word_count:
            getTotalFillerWords(),


        filler_words:
            fillerWords,


        /*
           Timestamp information
        */

        timestamps:
            transcriptSegments,


        /*
           Language
        */

        language:
            "English",


        language_code:
            languageSelect.value,


        /*
           Recording
        */

        recording: {

            available:
                Boolean(audioURL),

            mime_type:
                mediaRecorder
                ? mediaRecorder.mimeType
                : null

        }

    };

}



/* =====================================================
   26. DISPLAY BACKEND JSON
===================================================== */

function updateBackendJSON() {


    const data =
        createBackendData();


    jsonOutput.textContent =
        JSON.stringify(
            data,
            null,
            4
        );

}



/* =====================================================
   27. CLEAR BUTTON
===================================================== */

clearBtn.addEventListener(
    "click",
    function() {


        /*
           Do not clear while recording.
        */

        if (presentationActive) {

            alert(
                "Please stop the presentation before clearing the results."
            );

            return;

        }


        /*
           Clear transcript.
        */

        transcriptSegments = [];

        completeTranscript = "";

        temporaryTranscript = "";


        /*
           Clear pauses.
        */

        detectedPauses = [];


        /*
           Reset filler words.
        */

        fillerWords = {

            "um": 0,

            "uh": 0,

            "er": 0,

            "ah": 0,

            "actually": 0,

            "basically": 0,

            "like": 0,

            "you know": 0,

            "i mean": 0,

            "sort of": 0,

            "kind of": 0

        };


        /*
           Remove recording.
        */

        if (audioURL) {

            URL.revokeObjectURL(
                audioURL
            );

            audioURL = null;

        }


        audioPlayer.removeAttribute(
            "src"
        );

        audioPlayer.load();


        recordingInfo.textContent =
            "No recording available.";


        /*
           Reset timer.
        */

        presentationStartTime = null;

        finalSpeakingTime = 0;

        totalPausedTime = 0;

        manualPauseStart = null;


        /*
           Reset display.
        */

        speakingTime.textContent =
            "00:00";

        wpmDisplay.textContent =
            "0";

        pauseCountDisplay.textContent =
            "0";

        fillerCountDisplay.textContent =
            "0";


        /*
           Status.
        */

        statusDisplay.textContent =
            "● Ready";

        statusDisplay.className =
            "status ready";


        displayTranscript();

        displayFillerWords();

        displayPauses();

        updateBackendJSON();

    }
);



/* =====================================================
   28. BUTTON EVENTS
===================================================== */

startBtn.addEventListener(
    "click",
    startPresentation
);


pauseBtn.addEventListener(
    "click",
    pausePresentation
);


resumeBtn.addEventListener(
    "click",
    resumePresentation
);


stopBtn.addEventListener(
    "click",
    stopPresentation
);



/* =====================================================
   29. LANGUAGE CHANGE
===================================================== */

languageSelect.addEventListener(
    "change",
    function() {


        if (presentationActive) {

            alert(
                "Stop the current presentation before changing the language."
            );


            /*
               Restore English US if changed
               during presentation.
            */

            languageSelect.value =
                "en-US";

        }

    }
);



/* =====================================================
   30. INITIAL STATE
===================================================== */

displayTranscript();

displayFillerWords();

displayPauses();

updateBackendJSON();