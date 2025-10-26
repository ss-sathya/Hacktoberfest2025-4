// Compile: g++ -std=c++17 -O2 aqi_predictor.cpp -o aqi_predictor
// Run: ./aqi_predictor

#include <bits/stdc++.h>
using namespace std;

// ------- Data structures -------
struct Sample {
    double Temperature;
    double Humidity;
    double CO2;
    double PM2_5;
    double PM10;
    double NO2;
    double O3;
    double WindSpeed;
    int CityType; // 0 = Rural, 1 = Urban

    int AQI_Level;    // 0 Good, 1 Moderate, 2 Unhealthy, 3 Hazardous
    int Health_Risk;  // 0 Low, 1 Medium, 2 High
};

// ------- Random real in [a,b] -------
double rnd_double(double a, double b, std::mt19937 &rng) {
    std::uniform_real_distribution<double> dist(a, b);
    return dist(rng);
}

int rnd_int(int a, int b, std::mt19937 &rng) {
    std::uniform_int_distribution<int> dist(a, b);
    return dist(rng);
}

// ------- Labeling logic (same intuition as Python example) -------
int compute_aqi_level(const Sample &s) {
    // Weighted score of pollutants
    double score = 0.3 * s.PM2_5 + 0.2 * s.PM10 + 0.1 * s.NO2 + 0.05 * s.O3;
    if (score < 50) return 0;
    else if (score < 100) return 1;
    else if (score < 200) return 2;
    else return 3;
}

int compute_health_risk(const Sample &s) {
    int aqi = s.AQI_Level;
    if (aqi == 0) return 0;
    else if (aqi == 1) return 1;
    else if (aqi == 2 && s.CityType == 1) return 2;
    else {
        // for hazardous or tricky cases, bias toward higher risk in urban areas
        if (aqi == 3) return 2;
        return (s.CityType == 1) ? 2 : 1;
    }
}

// ------- Metrics -------
struct Metrics {
    double precision;
    double recall;
    double f1;
    int support;
};

Metrics compute_metrics_for_label(const vector<int> &y_true, const vector<int> &y_pred, int label) {
    int tp = 0, fp = 0, fn = 0;
    int n = y_true.size();
    for (int i = 0; i < n; ++i) {
        bool t = (y_true[i] == label);
        bool p = (y_pred[i] == label);
        if (t && p) tp++;
        if (!t && p) fp++;
        if (t && !p) fn++;
    }
    Metrics m{};
    m.support = 0;
    for (int v : y_true) if (v == label) m.support++;
    m.precision = (tp + fp) ? (double)tp / (tp + fp) : 0.0;
    m.recall = (tp + fn) ? (double)tp / (tp + fn) : 0.0;
    m.f1 = (m.precision + m.recall) ? 2.0 * m.precision * m.recall / (m.precision + m.recall) : 0.0;
    return m;
}

// ------- Main flow -------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // RNG setup
    random_device rd;
    mt19937 rng(rd());

    // 1) Generate synthetic dataset
    const int N = 1500;
    vector<Sample> data;
    data.reserve(N);

    for (int i = 0; i < N; ++i) {
        Sample s;
        s.Temperature = rnd_double(10.0, 45.0, rng);
        s.Humidity = rnd_double(20.0, 90.0, rng);
        s.CO2 = rnd_double(300.0, 800.0, rng);
        s.PM2_5 = rnd_double(5.0, 250.0, rng);
        s.PM10 = rnd_double(10.0, 300.0, rng);
        s.NO2 = rnd_double(2.0, 200.0, rng);
        s.O3 = rnd_double(5.0, 180.0, rng);
        s.WindSpeed = rnd_double(0.5, 10.0, rng);
        s.CityType = rnd_int(0, 1, rng);

        s.AQI_Level = compute_aqi_level(s);
        s.Health_Risk = compute_health_risk(s);

        data.push_back(s);
    }

    // 2) Shuffle and split train/test (we won't train, but keep for evaluation)
    shuffle(data.begin(), data.end(), rng);
    int train_size = (int)(0.8 * N);
    vector<Sample> train(data.begin(), data.begin() + train_size);
    vector<Sample> test(data.begin() + train_size, data.end());

    // 3) "Predictor" — in this example we simulate a real predictor by applying the same rule
    //     (this emulates a deterministic labeler; in a real ML flow you'd train a model)
    vector<int> y_true_aqi, y_pred_aqi;
    vector<int> y_true_hr, y_pred_hr;
    for (auto &s : test) {
        y_true_aqi.push_back(s.AQI_Level);
        y_true_hr.push_back(s.Health_Risk);

        // As a baseline predictor, apply the same deterministic function (idealized)
        int pred_aqi = compute_aqi_level(s);
        int pred_hr = compute_health_risk(s);

        y_pred_aqi.push_back(pred_aqi);
        y_pred_hr.push_back(pred_hr);
    }

    // 4) Evaluate
    cout << "🌫️ Air Quality & Health Prediction (C++ Rule-based Simulator)\n\n";

    // Calculate overall accuracy
    int correct_aqi = 0;
    for (size_t i = 0; i < y_true_aqi.size(); ++i) if (y_true_aqi[i] == y_pred_aqi[i]) correct_aqi++;
    double acc_aqi = (double)correct_aqi / y_true_aqi.size();

    int correct_hr = 0;
    for (size_t i = 0; i < y_true_hr.size(); ++i) if (y_true_hr[i] == y_pred_hr[i]) correct_hr++;
    double acc_hr = (double)correct_hr / y_true_hr.size();

    cout << "Overall Accuracy:\n";
    cout << "  AQI_Level Accuracy: " << fixed << setprecision(3) << acc_aqi << "\n";
    cout << "  Health_Risk Accuracy: " << fixed << setprecision(3) << acc_hr << "\n\n";

    // Per-class metrics for AQI (labels 0..3)
    cout << "AQI_Level Classification Report:\n";
    cout << "Label  Precision  Recall   F1-score  Support\n";
    for (int lbl = 0; lbl <= 3; ++lbl) {
        Metrics m = compute_metrics_for_label(y_true_aqi, y_pred_aqi, lbl);
        cout << setw(5) << lbl << "  "
             << setw(9) << setprecision(3) << m.precision << "  "
             << setw(6) << setprecision(3) << m.recall << "  "
             << setw(8) << setprecision(3) << m.f1 << "  "
             << setw(7) << m.support << "\n";
    }
    cout << "\n";

    // Per-class metrics for Health Risk (0..2)
    cout << "Health_Risk Classification Report:\n";
    cout << "Label  Precision  Recall   F1-score  Support\n";
    for (int lbl = 0; lbl <= 2; ++lbl) {
        Metrics m = compute_metrics_for_label(y_true_hr, y_pred_hr, lbl);
        cout << setw(5) << lbl << "  "
             << setw(9) << setprecision(3) << m.precision << "  "
             << setw(6) << setprecision(3) << m.recall << "  "
             << setw(8) << setprecision(3) << m.f1 << "  "
             << setw(7) << m.support << "\n";
    }
    cout << "\n";

    // 5) Interactive single-sample prediction
    cout << "Enter a custom sample to predict (or type 'demo' to run a demo sample):\n";
    cout << "Format: Temperature Humidity CO2 PM2.5 PM10 NO2 O3 WindSpeed CityType(0/1)\n";
    cout << "Example: 33 65 550 150 180 80 60 3.5 1\n";
    cout << "Input: ";

    string line;
    getline(cin, line);
    if (line.empty()) getline(cin, line); // handle newline from previous input

    Sample s;
    bool use_demo = false;
    if (line == "demo") {
        use_demo = true;
        s.Temperature = 33;
        s.Humidity = 65;
        s.CO2 = 550;
        s.PM2_5 = 150;
        s.PM10 = 180;
        s.NO2 = 80;
        s.O3 = 60;
        s.WindSpeed = 3.5;
        s.CityType = 1;
    } else {
        // parse
        istringstream iss(line);
        if (!(iss >> s.Temperature >> s.Humidity >> s.CO2 >> s.PM2_5 >> s.PM10 >> s.NO2 >> s.O3 >> s.WindSpeed >> s.CityType)) {
            cout << "Invalid input. Running demo sample.\n";
            use_demo = true;
            s.Temperature = 33;
            s.Humidity = 65;
            s.CO2 = 550;
            s.PM2_5 = 150;
            s.PM10 = 180;
            s.NO2 = 80;
            s.O3 = 60;
            s.WindSpeed = 3.5;
            s.CityType = 1;
        }
    }

    s.AQI_Level = compute_aqi_level(s);
    s.Health_Risk = compute_health_risk(s);

    static const char* AQI_labels[] = {"Good", "Moderate", "Unhealthy", "Hazardous"};
    static const char* HR_labels[] = {"Low", "Medium", "High"};

    cout << "\nPrediction for the sample:\n";
    cout << "  AQI Score -> Level " << s.AQI_Level << " (" << AQI_labels[s.AQI_Level] << ")\n";
    cout << "  Health Risk -> " << s.Health_Risk << " (" << HR_labels[s.Health_Risk] << ")\n";

    cout << "\nDone. You can adapt this program to load real CSV data, train a model,\n"
         << "or export the synthetic data for downstream ML experimentation.\n";

    return 0;
}
