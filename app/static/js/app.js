/**
 * Plant & Fruit Classifier - Frontend Client Logic
 * Designed for responsive Web App matching Project Slides & Documentation Wireframe.
 */

document.addEventListener("DOMContentLoaded", () => {
    // State variables
    let selectedFile = null;
    let selectedDataUrl = null;
    let currentMode = "disease";
    let webcamStream = null;

    // DOM Elements
    const fileInput = document.getElementById("file-input");
    const dropZone = document.getElementById("drop-zone");
    const browseBtn = document.getElementById("browse-btn");
    const webcamBtn = document.getElementById("webcam-btn");

    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const fileThumb = document.getElementById("file-thumb");
    const fileName = document.getElementById("file-name");
    const fileSize = document.getElementById("file-size");
    const clearImageBtn = document.getElementById("clear-image-btn");
    const analyzeBtn = document.getElementById("analyze-btn");
    const scanAnimation = document.getElementById("scan-animation");

    // Result Elements
    const resultPlaceholder = document.getElementById("result-placeholder");
    const resultLoading = document.getElementById("result-loading");
    const resultError = document.getElementById("result-error");
    const errorTitle = document.getElementById("error-title");
    const errorMessage = document.getElementById("error-message");
    const retryBtn = document.getElementById("retry-btn");
    const resultContent = document.getElementById("result-content");

    const resultImageThumb = document.getElementById("result-image-thumb");
    const cropBadge = document.getElementById("crop-badge");
    const healthStatusBadge = document.getElementById("health-status-badge");
    const diseaseNameTh = document.getElementById("disease-name-th");
    const diseaseNameEn = document.getElementById("disease-name-en");
    const confidencePercentage = document.getElementById("confidence-percentage");
    const confidenceBarFill = document.getElementById("confidence-bar-fill");

    const diseaseSymptoms = document.getElementById("disease-symptoms");
    const diseaseCauses = document.getElementById("disease-causes");
    const diseaseTreatmentList = document.getElementById("disease-treatment-list");
    const diseasePreventionList = document.getElementById("disease-prevention-list");
    const topPredictionsContainer = document.getElementById("top-predictions-container");

    // Navigation & Modals
    const modeTabs = document.querySelectorAll(".mode-tab");
    const sampleBtns = document.querySelectorAll(".sample-btn");
    const historyModal = document.getElementById("history-modal");
    const openHistoryBtn = document.getElementById("open-history-btn");
    const closeHistoryBtn = document.getElementById("close-history-btn");
    const clearHistoryBtn = document.getElementById("clear-history-btn");
    const historyList = document.getElementById("history-list");
    const historyCounter = document.getElementById("history-counter");

    const aboutModal = document.getElementById("about-modal");
    const openAboutBtn = document.getElementById("open-about-btn");
    const closeAboutBtn = document.getElementById("close-about-btn");

    // Webcam Modal Elements
    const webcamModal = document.getElementById("webcam-modal");
    const webcamVideo = document.getElementById("webcam-video");
    const webcamCanvas = document.getElementById("webcam-canvas");
    const closeWebcamBtn = document.getElementById("close-webcam-btn");
    const cancelWebcamBtn = document.getElementById("cancel-webcam-btn");
    const captureWebcamBtn = document.getElementById("capture-webcam-btn");

    // 1. Health check & system status
    fetch("/api/health")
        .then(res => res.json())
        .then(data => {
            const statusEl = document.getElementById("system-device");
            if (data && data.device) {
                statusEl.textContent = data.device;
            }
        })
        .catch(() => {
            const statusEl = document.getElementById("system-device");
            if (statusEl) statusEl.textContent = "ResNet50 AI Active";
        });

    // 2. Mode Tabs selection
    modeTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            modeTabs.forEach(t => {
                t.classList.remove("active-tab");
                t.classList.add("inactive-tab");
            });
            tab.classList.remove("inactive-tab");
            tab.classList.add("active-tab");
            currentMode = tab.getAttribute("data-mode") || "disease";

            if (selectedFile) {
                runAnalysis();
            }
        });
    });

    // 3. File Browse & Dropzone
    browseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener("click", () => {
        fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag and Drop
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("border-emerald-500", "bg-emerald-50/40");
        }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("border-emerald-500", "bg-emerald-50/40");
        }, false);
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files.length > 0) {
            handleFileSelect(dt.files[0]);
        }
    });

    function formatBytes(bytes, decimals = 1) {
        if (!+bytes) return "0 Bytes";
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
    }

    function handleFileSelect(file) {
        if (!file) return;

        // Validation for image file type
        const isValid = file.type.startsWith("image/") || /\.(jpg|jpeg|png|webp|jfif|bmp)$/i.test(file.name);
        if (!isValid) {
            showError("รูปแบบไฟล์ไม่ถูกต้อง", "กรุณาอัปโหลดไฟล์รูปภาพที่รองรับเท่านั้น (.jpg, .jpeg, .png, .webp)");
            return;
        }

        selectedFile = file;
        fileName.textContent = file.name;
        fileSize.textContent = `(${formatBytes(file.size)})`;

        const reader = new FileReader();
        reader.onload = (e) => {
            selectedDataUrl = e.target.result;
            imagePreview.src = selectedDataUrl;
            fileThumb.src = selectedDataUrl;
            previewContainer.classList.remove("hidden");
            // Smooth scroll to preview on mobile
            if (window.innerWidth < 768) {
                previewContainer.scrollIntoView({ behavior: "smooth" });
            }
        };
        reader.readAsDataURL(file);
    }

    // 4. Sample Images Click (Instant testing with local high-quality images)
    sampleBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const filename = btn.getAttribute("data-sample");
            const sampleUrl = `/static/images/samples/${filename}`;

            fetch(sampleUrl)
                .then(res => {
                    if (!res.ok) throw new Error("Could not load sample image");
                    return res.blob();
                })
                .then(blob => {
                    const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
                    handleFileSelect(file);
                    // Automatic trigger analysis on sample click
                    setTimeout(() => {
                        runAnalysis();
                    }, 100);
                })
                .catch(err => {
                    console.error("Sample load error:", err);
                    showError("ไม่สามารถโหลดตัวอย่างภาพได้", "กรุณาลองเลือกภาพอื่น หรืออัปโหลดจากเครื่องของคุณ");
                });
        });
    });

    // 5. Clear image
    clearImageBtn.addEventListener("click", () => {
        selectedFile = null;
        selectedDataUrl = null;
        fileInput.value = "";
        imagePreview.src = "";
        fileThumb.src = "";
        previewContainer.classList.add("hidden");
        resetResultView();
    });

    // 6. Main Analyze button & logic
    analyzeBtn.addEventListener("click", () => {
        runAnalysis();
    });

    retryBtn.addEventListener("click", () => {
        if (selectedFile) {
            runAnalysis();
        } else {
            resetResultView();
        }
    });

    function resetResultView() {
        resultPlaceholder.classList.remove("hidden");
        resultLoading.classList.add("hidden");
        resultError.classList.add("hidden");
        resultContent.classList.add("hidden");
        scanAnimation.classList.add("hidden");
    }

    function showError(title, message) {
        resultPlaceholder.classList.add("hidden");
        resultLoading.classList.add("hidden");
        resultContent.classList.add("hidden");
        scanAnimation.classList.add("hidden");
        resultError.classList.remove("hidden");

        errorTitle.textContent = title;
        errorMessage.textContent = message;
    }

    function runAnalysis() {
        if (!selectedFile) {
            alert("กรุณาเลือกหรืออัปโหลดรูปภาพก่อนทำการวิเคราะห์");
            return;
        }

        // Show loading state
        resultPlaceholder.classList.add("hidden");
        resultError.classList.add("hidden");
        resultContent.classList.add("hidden");
        resultLoading.classList.remove("hidden");
        scanAnimation.classList.remove("hidden");

        // Send FormData to /api/predict
        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("mode", currentMode);

        fetch("/api/predict", {
            method: "POST",
            body: formData
        })
        .then(async res => {
            const data = await res.json();
            if (!res.ok) {
                throw new Error(data.detail || data.error || `HTTP ${res.status}`);
            }
            return data;
        })
        .then(data => {
            displayResults(data);
            saveToHistory(data);
        })
        .catch(err => {
            console.error("Inference Error:", err);
            showError("เกิดข้อผิดพลาดในการวิเคราะห์", err.message || "ไม่สามารถเชื่อมต่อกับเซิร์ฟเวอร์ AI ได้ กรุณาลองใหม่อีกครั้ง");
        })
        .finally(() => {
            scanAnimation.classList.add("hidden");
        });
    }

    function displayResults(data) {
        // Result fields can come directly or under data.prediction
        const pred = data.prediction || data;

        resultLoading.classList.add("hidden");
        resultError.classList.add("hidden");
        resultContent.classList.remove("hidden");

        // Thumbnail
        resultImageThumb.src = selectedDataUrl || imagePreview.src;

        // Crop Badge
        const cropText = pred.crop || data.crop || "พืช / ผลไม้";
        cropBadge.textContent = cropText;

        // Health Status Badge
        const isHealthy = pred.is_healthy !== undefined ? pred.is_healthy : (data.is_healthy || false);
        if (pred.mode === "ripeness" || data.mode === "ripeness") {
            if ((pred.predicted_class || "").includes("Rotten")) {
                healthStatusBadge.className = "inline-flex items-center gap-1 px-3 py-0.5 rounded-full text-xs font-semibold bg-rose-100 text-rose-800 border border-rose-200";
                healthStatusBadge.innerHTML = `<i data-lucide="alert-octagon" class="w-3.5 h-3.5"></i> เน่าเสีย`;
            } else {
                healthStatusBadge.className = "inline-flex items-center gap-1 px-3 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200";
                healthStatusBadge.innerHTML = `<i data-lucide="check-circle" class="w-3.5 h-3.5"></i> คุณภาพดี / ปลอดภัย`;
            }
        } else {
            if (isHealthy) {
                healthStatusBadge.className = "inline-flex items-center gap-1 px-3 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200";
                healthStatusBadge.innerHTML = `<i data-lucide="check-circle" class="w-3.5 h-3.5"></i> สุขภาพดี / ปกติ`;
            } else {
                healthStatusBadge.className = "inline-flex items-center gap-1 px-3 py-0.5 rounded-full text-xs font-semibold bg-amber-100 text-amber-800 border border-amber-200";
                healthStatusBadge.innerHTML = `<i data-lucide="alert-triangle" class="w-3.5 h-3.5"></i> พบโรคพืช`;
            }
        }

        // Title and English Name
        const nameTh = pred.thai_name || pred.name_th || data.thai_name || "ไม่ทราบชื่อโรค";
        const nameEn = pred.name_en || data.name_en || "";
        diseaseNameTh.textContent = nameTh;
        diseaseNameEn.textContent = nameEn ? `(${nameEn})` : "";

        // Confidence Percentage
        const confVal = parseFloat(pred.confidence || data.confidence || 0);
        confidencePercentage.textContent = `${confVal.toFixed(2)}%`;
        confidenceBarFill.style.width = `${Math.min(100, Math.max(10, confVal))}%`;

        // Symptoms & Causes
        diseaseSymptoms.textContent = pred.symptoms || data.symptoms || "ไม่มีข้อมูลอาการระบุไว้";
        diseaseCauses.textContent = pred.causes || data.causes || "ไม่มีข้อมูลสาเหตุระบุไว้";

        // Treatment (Numbered badges 1, 2, 3, 4)
        diseaseTreatmentList.innerHTML = "";
        let treatments = pred.treatment || data.treatment || [];
        if (typeof treatments === "string") {
            treatments = treatments.split("\n").filter(t => t.trim().length > 0);
        } else if (!treatments || treatments.length === 0) {
            if (pred.recommendation || data.recommendation) {
                treatments = (pred.recommendation || data.recommendation).split("\n").filter(t => t.trim().length > 0);
            }
        }

        if (treatments.length === 0) {
            treatments = ["หมั่นดูแลรดน้ำและตรวจสอบแปลงอย่างสม่ำเสมอ"];
        }

        treatments.forEach((item, idx) => {
            // Strip any existing "1. " or "1 " leading text if present
            const cleanText = item.replace(/^\d+[\.\s\-]+\s*/, "").trim();
            const li = document.createElement("li");
            li.className = "flex items-start gap-2.5";
            li.innerHTML = `
                <span class="badge-number">${idx + 1}</span>
                <span class="pt-0.5">${cleanText}</span>
            `;
            diseaseTreatmentList.appendChild(li);
        });

        // Prevention Tips
        diseasePreventionList.innerHTML = "";
        let preventions = pred.prevention || data.prevention || [];
        if (typeof preventions === "string") {
            preventions = preventions.split("\n").filter(p => p.trim().length > 0);
        }
        if (preventions.length === 0) {
            preventions = ["รักษาสุขอนามัยในแปลงปลูกอย่างเคร่งครัดเพื่อลดความเสี่ยง"];
        }

        preventions.forEach(item => {
            const cleanText = item.replace(/^\d+[\.\s\-]+\s*/, "").trim();
            const li = document.createElement("li");
            li.className = "flex items-start gap-2";
            li.innerHTML = `
                <i data-lucide="check" class="w-4 h-4 text-sky-600 mt-0.5 flex-shrink-0"></i>
                <span>${cleanText}</span>
            `;
            diseasePreventionList.appendChild(li);
        });

        // Probability Breakdown
        topPredictionsContainer.innerHTML = "";
        const topList = pred.top_predictions || data.top_predictions || [];
        topList.slice(0, 4).forEach(item => {
            const label = item.name_th || item.label || item.class || "";
            const score = parseFloat(item.confidence || item.score || 0);

            const row = document.createElement("div");
            row.className = "space-y-1";
            row.innerHTML = `
                <div class="flex items-center justify-between text-xs text-slate-700">
                    <span class="font-medium">${label}</span>
                    <span class="font-mono font-semibold">${score.toFixed(1)}%</span>
                </div>
                <div class="w-full bg-slate-200/70 rounded-full h-1.5 overflow-hidden">
                    <div class="bg-emerald-600 h-1.5 rounded-full" style="width: ${Math.min(100, Math.max(2, score))}%"></div>
                </div>
            `;
            topPredictionsContainer.appendChild(row);
        });

        // Re-initialize Lucide Icons
        if (window.lucide) {
            window.lucide.createIcons();
        }
    }

    // 7. Local Storage History Management
    function saveToHistory(data) {
        try {
            const history = JSON.parse(localStorage.getItem("plant_history") || "[]");
            const pred = data.prediction || data;
            const record = {
                id: Date.now(),
                timestamp: new Date().toLocaleTimeString("th-TH", { hour: "2-digit", minute: "2-digit", day: "numeric", month: "short" }),
                name_th: pred.thai_name || pred.name_th || "การวิเคราะห์โรคพืช",
                name_en: pred.name_en || "",
                crop: pred.crop || "พืช / ผลไม้",
                confidence: pred.confidence || 0,
                thumb: selectedDataUrl,
                data: data
            };
            history.unshift(record);
            if (history.length > 20) history.pop(); // keep last 20
            localStorage.setItem("plant_history", JSON.stringify(history));
            updateHistoryCounter();
        } catch (e) {
            console.warn("Could not save to history:", e);
        }
    }

    function updateHistoryCounter() {
        try {
            const history = JSON.parse(localStorage.getItem("plant_history") || "[]");
            if (historyCounter) {
                historyCounter.textContent = history.length;
            }
        } catch (e) {}
    }

    function renderHistoryList() {
        try {
            const history = JSON.parse(localStorage.getItem("plant_history") || "[]");
            historyList.innerHTML = "";

            if (history.length === 0) {
                historyList.innerHTML = `
                    <div class="text-center py-10 text-slate-400">
                        <i data-lucide="inbox" class="w-10 h-10 mx-auto mb-2 text-slate-300"></i>
                        <p class="text-xs">ยังไม่มีประวัติการวิเคราะห์</p>
                    </div>
                `;
                if (window.lucide) window.lucide.createIcons();
                return;
            }

            history.forEach(item => {
                const el = document.createElement("div");
                el.className = "flex items-center justify-between p-3 rounded-2xl border border-slate-200 hover:bg-emerald-50/40 cursor-pointer transition";
                el.innerHTML = `
                    <div class="flex items-center space-x-3 overflow-hidden">
                        <img src="${item.thumb || '/static/images/samples/leaf_sample_01.jpg'}" class="w-12 h-12 rounded-xl object-cover border border-slate-200 flex-shrink-0">
                        <div class="truncate">
                            <h5 class="text-xs font-bold text-slate-800 truncate">${item.name_th}</h5>
                            <p class="text-[11px] text-slate-400">${item.timestamp} • ความเชื่อมั่น ${item.confidence.toFixed(1)}%</p>
                        </div>
                    </div>
                    <span class="text-xs text-emerald-600 font-semibold flex items-center gap-1 flex-shrink-0">
                        ดูผล <i data-lucide="chevron-right" class="w-3.5 h-3.5"></i>
                    </span>
                `;
                el.addEventListener("click", () => {
                    selectedDataUrl = item.thumb;
                    imagePreview.src = item.thumb;
                    fileThumb.src = item.thumb;
                    fileName.textContent = item.name_th;
                    fileSize.textContent = "(ประวัติย้อนหลัง)";
                    previewContainer.classList.remove("hidden");
                    displayResults(item.data);
                    historyModal.classList.add("hidden");
                });
                historyList.appendChild(el);
            });

            if (window.lucide) window.lucide.createIcons();
        } catch (e) {
            console.error(e);
        }
    }

    // Modal Listeners
    if (openHistoryBtn) {
        openHistoryBtn.addEventListener("click", () => {
            renderHistoryList();
            historyModal.classList.remove("hidden");
        });
    }

    if (closeHistoryBtn) {
        closeHistoryBtn.addEventListener("click", () => {
            historyModal.classList.add("hidden");
        });
    }

    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener("click", () => {
            if (confirm("คุณต้องการล้างประวัติการวิเคราะห์ทั้งหมดหรือไม่?")) {
                localStorage.removeItem("plant_history");
                renderHistoryList();
                updateHistoryCounter();
            }
        });
    }

    if (openAboutBtn) {
        openAboutBtn.addEventListener("click", () => {
            aboutModal.classList.remove("hidden");
        });
    }

    if (closeAboutBtn) {
        closeAboutBtn.addEventListener("click", () => {
            aboutModal.classList.add("hidden");
        });
    }

    [historyModal, aboutModal].forEach(modal => {
        if (modal) {
            modal.addEventListener("click", (e) => {
                if (e.target === modal) {
                    modal.classList.add("hidden");
                }
            });
        }
    });

    // 8. Webcam Implementation
    if (webcamBtn) {
        webcamBtn.addEventListener("click", async () => {
            try {
                webcamStream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: "environment" }
                });
                webcamVideo.srcObject = webcamStream;
                webcamModal.classList.remove("hidden");
            } catch (err) {
                console.error("Webcam error:", err);
                alert("ไม่สามารถเข้าถึงกล้องได้ กรุณาตรวจสอบการอนุญาตใช้งานกล้องในเบราว์เซอร์");
            }
        });
    }

    function stopWebcam() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        webcamModal.classList.add("hidden");
    }

    if (closeWebcamBtn) closeWebcamBtn.addEventListener("click", stopWebcam);
    if (cancelWebcamBtn) cancelWebcamBtn.addEventListener("click", stopWebcam);

    if (captureWebcamBtn) {
        captureWebcamBtn.addEventListener("click", () => {
            if (!webcamVideo.videoWidth) return;
            webcamCanvas.width = webcamVideo.videoWidth;
            webcamCanvas.height = webcamVideo.videoHeight;
            const ctx = webcamCanvas.getContext("2d");
            ctx.drawImage(webcamVideo, 0, 0);

            webcamCanvas.toBlob(blob => {
                const capturedFile = new File([blob], `capture_${Date.now()}.jpg`, { type: "image/jpeg" });
                handleFileSelect(capturedFile);
                stopWebcam();
                setTimeout(() => {
                    runAnalysis();
                }, 200);
            }, "image/jpeg", 0.95);
        });
    }

    // Initial load
    updateHistoryCounter();
});
