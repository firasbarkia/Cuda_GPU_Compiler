let editor;

require.config({ paths: { vs: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.36.1/min/vs' } });

require(['vs/editor/editor.main'], function () {
    editor = monaco.editor.create(document.getElementById('editor-container'), {
        value: `#include <iostream>
#include <cuda_runtime.h>

__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d!\\n", threadIdx.x);
}

int main() {
    printf("Hello World from CPU!\\n");

    helloFromGPU<<<1, 5>>>();
    
    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}`,
        language: 'cpp',
        theme: 'vs-dark',
        automaticLayout: true,
        fontSize: 14,
        fontFamily: "'Fira Code', monospace",
        minimap: { enabled: false },
        lineNumbers: 'on',
        roundedSelection: true,
        scrollBeyondLastLine: false,
        readOnly: false,
        cursorStyle: 'line',
        padding: { top: 20 }
    });
});

// Tab Switching
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

        btn.classList.add('active');
        const tabId = btn.getAttribute('data-tab') + '-tab';
        document.getElementById(tabId).classList.add('active');
    });
});

// Run Code
document.getElementById('runBtn').addEventListener('click', async () => {
    const code = editor.getValue();
    const loader = document.getElementById('loader');
    const consoleOutput = document.getElementById('console-output');
    const logsOutput = document.getElementById('logs-output');
    const ptxOutput = document.getElementById('ptx-output');
    const sassOutput = document.getElementById('sass-output');
    const logsTabBtn = document.querySelector('[data-tab="logs"]');

    loader.classList.remove('hidden');
    logsTabBtn.classList.remove('has-error');
    logsOutput.classList.remove('error');

    try {
        const response = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ code })
        });

        const data = await response.json();

        // Update UI
        consoleOutput.textContent = data.output || (data.success ? "Success (no output)" : "// Executable failed or crashed");
        logsOutput.textContent = data.error || (data.success ? "Compilation successful!" : "");

        ptxOutput.textContent = data.emissions.ptx || "// PTX empty";
        sassOutput.textContent = data.emissions.sass || "// SASS empty";

        // Logic to switch tabs based on success/error
        if (!data.success) {
            logsTabBtn.classList.add('has-error');
            logsOutput.classList.add('error');
            logsTabBtn.click(); // Automatic switch to logs on error
        } else {
            document.querySelector('[data-tab="output"]').click(); // Switch to output on success
        }

    } catch (err) {
        consoleOutput.textContent = "Fatal Error connecting to server: " + err.message;
    } finally {
        loader.classList.add('hidden');
    }
});
