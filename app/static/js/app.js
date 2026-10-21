/**
 * Plant & Fruit Disease Classifier - Frontend Client Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    // State variables
    let selectedFile = null;
    let currentMode = "disease";
    let webcamStream = null;

    // DOM Elements
    const fileInput = document.getElementById("file-input");
    const dropZone = document.getElementById("drop-zone");
    const browseBtn = document.getElementById("browse-btn");
    const webcamBtn = document.getElementById("webcam-btn");
    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const clearImageBtn = document.getElementById("clear-image-btn");
    const analyzeBtn = document.getElementById("analyze-btn");
    const scanAnimation = document.getElementById("scan-animation");

    const resultPlaceholder = document.getElementById("result-placeholder");
    const resultLoading = document.getElementById("result-loading");
    const resultError = document.getElementById("result-error");
    const errorTitle = document.getElementById("error-title");
    const errorMessage = document.getElementById("error-message");
    const retryBtn = document.getElementById("retry-btn");
    const resultContent = document.getElementById("result-content");

    const modeTabs = document.querySelectorAll(".mode-tab");
    const sampleBtns = document.querySelectorAll(".sample-btn");

    // Result card fields
    const cropBadge = document.getElementById("crop-badge");
    const diseaseNameTh = document.getElementById("disease-name-th");
    const diseaseNameEn = document.getElementById("disease-name-en");
    const healthStatusBadge = document.getElementById("health-status-badge");
    const confidencePercentage = document.getElementById("confidence-percentage");
    const confidenceBarFill = document.getElementById("confidence-bar-fill");
    const statusAccentBar = document.getElementById("status-accent-bar");

    const diseaseSymptoms = document.getElementById("disease-symptoms");
    const diseaseCauses = document.getElementById("disease-causes");
    const diseaseTreatmentList = document.getElementById("disease-treatment-list");
    const diseasePreventionList = document.getElementById("disease-prevention-list");
    const topPredictionsContainer = document.getElementById("top-predictions-container");

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
            if (data.cuda_available) {
                statusEl.textContent = `GPU: ${data.device}`;
            } else {
                statusEl.textContent = `CPU Mode (${data.pytorch_version})`;
            }
        })
        .catch(() => {
            const statusEl = document.getElementById("system-device");
            statusEl.textContent = "AI Model Engine Ready";
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
            currentMode = tab.getAttribute("data-mode");

            // If an image is already loaded, re-analyze
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
        if (e.target.files && e.target.files[0]) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // Drag & Drop events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("border-brand-500", "bg-emerald-50/40");
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("border-brand-500", "bg-emerald-50/40");
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files && files.length > 0) {
            handleSelectedFile(files[0]);
        }
    });

    function handleSelectedFile(file) {
        if (!file.type.startsWith("image/")) {
            alert("กรุณาเลือกไฟล์ที่เป็นรูปภาพเท่านั้น (.jpg, .png, .webp)");
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove("hidden");
            // Automatically analyze on selection
            runAnalysis();
        };
        reader.readAsDataURL(file);
    }

    // 4. Quick Sample Images
    sampleBtns.forEach(btn => {
        btn.addEventListener("click", async () => {
            const img = btn.querySelector("img");
            if (!img) return;

            try {
                // Fetch image from URL and convert to File
                const response = await fetch(img.src);
                const blob = await response.blob();
                const file = new File([blob], `${btn.getAttribute("data-sample")}.jpg`, { type: "image/jpeg" });
                handleSelectedFile(file);
            } catch (err) {
                console.error("Error loading sample image:", err);
                imagePreview.src = img.src;
                previewContainer.classList.remove("hidden");
            }
        });
    });

    // 5. Clear image
    clearImageBtn.addEventListener("click", () => {
        selectedFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        previewContainer.classList.add("hidden");
        resultContent.classList.add("hidden");
        if (resultError) resultError.classList.add("hidden");
        resultPlaceholder.classList.remove("hidden");
    });

    // 6. Manual Analyze & Retry Buttons
    analyzeBtn.addEventListener("click", () => {
        if (selectedFile) {
            runAnalysis();
        }
    });

    if (retryBtn) {
        retryBtn.addEventListener("click", () => {
            if (selectedFile) {
                runAnalysis();
            }
        });
    }

    /**
     * Resizes large images client-side before uploading.
     * Prevents high bandwidth usage, mobile upload timeouts, and backend 512MB RAM OOM.
     */
    async function prepareImageForUpload(file) {
        // If image is already smaller than 1.5MB and not HEIC, send as is
        if (file.size <= 1.5 * 1024 * 1024 && !file.type.includes("heic")) {
            return file;
        }

        return new Promise((resolve) => {
            const img = new Image();
            const objectUrl = URL.createObjectURL(file);
            img.onload = () => {
                URL.revokeObjectURL(objectUrl);
                const MAX_DIM = 1280;
                let width = img.width;
                let height = img.height;

                if (width > MAX_DIM || height > MAX_DIM) {
                    if (width > height) {
                        height = Math.round((height * MAX_DIM) / width);
                        width = MAX_DIM;
                    } else {
                        width = Math.round((width * MAX_DIM) / height);
                        height = MAX_DIM;
                    }
                }

                const canvas = document.createElement("canvas");
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext("2d");
                ctx.drawImage(img, 0, 0, width, height);

                canvas.toBlob((blob) => {
                    if (blob) {
                        const optimizedName = file.name.replace(/\.[^/.]+$/, "") + ".jpg";
                        const optimizedFile = new File([blob], optimizedName, {
                            type: "image/jpeg",
                            lastModified: Date.now()
                        });
                        resolve(optimizedFile);
                    } else {
                        resolve(file);
                    }
                }, "image/jpeg", 0.85);
            };
            img.onerror = () => {
                URL.revokeObjectURL(objectUrl);
                resolve(file);
            };
            img.src = objectUrl;
        });
    }

    // 7. Core Analysis Execution
    async function runAnalysis() {
        if (!selectedFile) return;

        // Reset display states
        resultPlaceholder.classList.add("hidden");
        resultContent.classList.add("hidden");
        if (resultError) resultError.classList.add("hidden");
        resultLoading.classList.remove("hidden");
        scanAnimation.classList.remove("hidden");

        try {
            // Compress large images client-side before upload
            const fileToSend = await prepareImageForUpload(selectedFile);

            const formData = new FormData();
            formData.append("file", fileToSend);
            formData.append("mode", currentMode);

            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                let errorMessageText = `เซิร์ฟเวอร์ตอบกลับรหัสข้อผิดพลาด (${response.status})`;
                try {
                    const contentType = response.headers.get("content-type") || "";
                    if (contentType.includes("application/json")) {
                        const errData = await response.json();
                        errorMessageText = errData.detail || errData.error || errorMessageText;
                    } else {
                        const text = await response.text();
                        if (response.status === 502 || response.status === 504 || response.status === 503) {
                            errorMessageText = "เซิร์ฟเวอร์ Render (Free Plan) กำลังเริ่มทำงานจากโหมดพัก (Cold Start) กรุณารอสักครู่ (ประมาณ 20-30 วินาที) แล้วกดปุ่ม 'ลองใหม่อีกครั้ง'";
                        } else if (text && text.trim().length > 0 && text.trim().length < 200 && !text.includes("<!DOCTYPE")) {
                            errorMessageText = text.trim();
                        }
                    }
                } catch (_) {
                    if (response.status === 502 || response.status === 504 || response.status === 503) {
                        errorMessageText = "เซิร์ฟเวอร์ Render (Free Plan) กำลังเริ่มทำงานจากโหมดพัก (Cold Start) กรุณารอสักครู่ (ประมาณ 20-30 วินาที) แล้วกดปุ่ม 'ลองใหม่อีกครั้ง'";
                    }
                }
                throw new Error(errorMessageText);
            }

            const contentType = response.headers.get("content-type") || "";
            if (!contentType.includes("application/json")) {
                throw new Error("เซิร์ฟเวอร์ส่งข้อมูลกลับมาในรูปแบบที่ไม่ถูกต้อง (กรุณากดลองใหม่อีกครั้ง)");
            }

            const data = await response.json();
            if (!data.success && data.error) {
                throw new Error(data.error);
            }

            renderResult(data.prediction);
        } catch (error) {
            console.error("Analysis error:", error);
            if (resultError && errorMessage) {
                errorMessage.textContent = error.message;
                resultError.classList.remove("hidden");
                if (window.lucide) lucide.createIcons();
            } else {
                alert(`ไม่สามารถวิเคราะห์ได้: ${error.message}`);
                resultPlaceholder.classList.remove("hidden");
            }
        } finally {
            resultLoading.classList.add("hidden");
            scanAnimation.classList.add("hidden");
        }
    }

    // 8. Render Result UI
    function renderResult(pred) {
        const info = pred.info;
        const isHealthy = pred.is_healthy;

        // Update basic headers
        cropBadge.textContent = info.crop || "พืช / ผลไม้";
        diseaseNameTh.textContent = info.name_th;
        diseaseNameEn.textContent = info.name_en;

        const conf = pred.confidence;
        confidencePercentage.textContent = `${conf}%`;
        confidenceBarFill.style.width = `${Math.min(100, Math.max(10, conf))}%`;

        // Status badge & accent coloring
        if (isHealthy) {
            healthStatusBadge.className = "inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 border border-emerald-200";
            healthStatusBadge.innerHTML = `<i data-lucide="check-circle" class="w-3.5 h-3.5"></i> สุขภาพดี / ปกติ`;
            statusAccentBar.className = "absolute top-0 left-0 right-0 h-1.5 bg-emerald-500";
            confidenceBarFill.className = "bg-emerald-600 h-2.5 rounded-full transition-all duration-700";
        } else {
            const isSevere = info.severity === "High";
            const badgeColor = isSevere ? "bg-rose-100 text-rose-800 border-rose-200" : "bg-amber-100 text-amber-800 border-amber-200";
            const iconName = isSevere ? "alert-triangle" : "alert-circle";
            const barColor = isSevere ? "bg-rose-500" : "bg-amber-500";
            
            healthStatusBadge.className = `inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${badgeColor}`;
            healthStatusBadge.innerHTML = `<i data-lucide="${iconName}" class="w-3.5 h-3.5"></i> พบอาการของโรค (${info.severity || 'ปานกลาง'})`;
            statusAccentBar.className = `absolute top-0 left-0 right-0 h-1.5 ${barColor}`;
            confidenceBarFill.className = `${barColor} h-2.5 rounded-full transition-all duration-700`;
        }

        // Details
        diseaseSymptoms.textContent = info.symptoms || "ไม่พบรายงานอาการผิดปกติที่เด่นชัด";
        diseaseCauses.textContent = info.causes || "ปัจจัยทางสภาพแวดล้อมหรือการจัดการแปลงปลูก";

        // Treatment List
        diseaseTreatmentList.innerHTML = "";
        if (info.treatment && Array.isArray(info.treatment)) {
            info.treatment.forEach(step => {
                const li = document.createElement("li");
                li.className = "flex items-start gap-2";
                li.innerHTML = `<i data-lucide="arrow-right-circle" class="w-4 h-4 text-emerald-600 mt-0.5 flex-shrink-0"></i> <span>${step}</span>`;
                diseaseTreatmentList.appendChild(li);
            });
        }

        // Prevention List
        diseasePreventionList.innerHTML = "";
        if (info.prevention && Array.isArray(info.prevention)) {
            info.prevention.forEach(step => {
                const li = document.createElement("li");
                li.className = "flex items-start gap-2";
                li.innerHTML = `<i data-lucide="shield" class="w-4 h-4 text-sky-600 mt-0.5 flex-shrink-0"></i> <span>${step}</span>`;
                diseasePreventionList.appendChild(li);
            });
        }

        // Top predictions probability breakdown
        topPredictionsContainer.innerHTML = "";
        if (pred.top_predictions && Array.isArray(pred.top_predictions)) {
            pred.top_predictions.forEach(item => {
                const label = item.name_th || item.label || item.class_key;
                const p = item.confidence || 0;

                const row = document.createElement("div");
                row.className = "text-xs space-y-1";
                row.innerHTML = `
                    <div class="flex justify-between text-slate-700 font-medium">
                        <span class="truncate pr-2">${label}</span>
                        <span>${p}%</span>
                    </div>
                    <div class="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                        <div class="bg-slate-400 h-1.5 rounded-full" style="width: ${Math.min(100, Math.max(3, p))}%"></div>
                    </div>
                `;
                topPredictionsContainer.appendChild(row);
            });
        }

        // Refresh Lucide icons inside dynamically created HTML
        if (window.lucide) {
            lucide.createIcons();
        }

        // Reveal result
        resultContent.classList.remove("hidden");
    }

    // 9. Live Webcam Integration
    webcamBtn.addEventListener("click", async () => {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } }
            });
            webcamVideo.srcObject = webcamStream;
            webcamModal.classList.remove("hidden");
        } catch (err) {
            alert("ไม่สามารถเข้าถึงกล้องเว็บแคมได้: " + err.message);
        }
    });

    function closeWebcam() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        webcamModal.classList.add("hidden");
    }

    closeWebcamBtn.addEventListener("click", closeWebcam);
    cancelWebcamBtn.addEventListener("click", closeWebcam);

    captureWebcamBtn.addEventListener("click", () => {
        if (!webcamVideo.videoWidth) return;

        webcamCanvas.width = webcamVideo.videoWidth;
        webcamCanvas.height = webcamVideo.videoHeight;
        const ctx = webcamCanvas.getContext("2d");
        ctx.drawImage(webcamVideo, 0, 0, webcamCanvas.width, webcamCanvas.height);

        webcamCanvas.toBlob(blob => {
            const capturedFile = new File([blob], `capture_${Date.now()}.jpg`, { type: "image/jpeg" });
            closeWebcam();
            handleSelectedFile(capturedFile);
        }, "image/jpeg", 0.95);
    });

});
