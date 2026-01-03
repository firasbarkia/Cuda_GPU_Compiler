const express = require('express');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();
app.use(express.json());
app.use(cors());
app.use(express.static('public'));

const WORKSPACE = '/workspace';

app.post('/api/run', (req, res) => {
    const { code } = req.body;

    // 1. Write code to file
    fs.writeFileSync(path.join(WORKSPACE, 'hello.cu'), code);

    // 2. Run Make and Execute
    // We run make clean first to ensure fresh emissions
    exec('make clean && make && ./hello', { cwd: WORKSPACE }, (error, stdout, stderr) => {
        const buildLog = stderr;
        const output = stdout;

        // 3. Read emissions if they exist
        let ptx = '';
        let sass = '';
        try {
            ptx = fs.readFileSync(path.join(WORKSPACE, 'hello.ptx'), 'utf8');
            sass = fs.readFileSync(path.join(WORKSPACE, 'hello.sass'), 'utf8');
        } catch (e) {
            console.log("Emissions not yet ready or failed");
        }

        res.json({
            success: !error,
            output: output,
            error: buildLog,
            emissions: {
                ptx: ptx,
                sass: sass
            }
        });
    });
});

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`CUDA Playground Server running on http://localhost:${PORT}`);
});
