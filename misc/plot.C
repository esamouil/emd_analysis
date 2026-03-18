#include <TGraph.h>
#include <TCanvas.h>
#include <TAxis.h>
#include <TApplication.h>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

int plot() {
    // ROOT application for interactivity
    int argc = 0;
    char* argv[] = {};
    TApplication app("app", &argc, argv);

    // Load CSV
    std::ifstream file("/home/esamouil/data_ess__/data_psi_aug_2024/DREAM_B/Z010824_0001.csv"); // <-- your CSV
    std::vector<double> timestamps;
    std::vector<double> adc_values;
    std::string line;

    // skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string ts_str, adc_str;
        if (std::getline(ss, ts_str, ',') && std::getline(ss, adc_str, ',')) {
            timestamps.push_back(std::stod(ts_str));
            adc_values.push_back(std::stod(adc_str));
        }
    }

    int n = timestamps.size();

    // Create TGraph
    TGraph* graph = new TGraph(n, &timestamps[0], &adc_values[0]);
    graph->SetTitle("ADC vs Timestamp;Timestamp [µs];ADC Value");
    graph->SetMarkerStyle(20);
    graph->SetMarkerColor(kBlue);

    // Canvas
    TCanvas* c1 = new TCanvas("c1", "Interactive ADC Plot", 800, 500);
    c1->SetGrid();
    graph->Draw("AP"); // A=axis, P=points

    c1->Update();

    // Keep the canvas open
    c1->Connect("CloseWindow()", "TApplication", &app, "Terminate()");
    app.Run();

    
    return 0;
}
