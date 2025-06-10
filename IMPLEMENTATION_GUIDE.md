# Dosha Quantification Model Implementation

This document provides an overview of the Dosha Quantification Model implementation and how to use it in the IridoVeda application.

## Implementation Summary

The Dosha Quantification Model has been fully implemented with the following components:

1. **Core Model**: `dosha_quantification_model.py` contains the complete implementation of the model that quantifies Vata, Pitta, and Kapha doshas from iris images and connects them to health metrics and organ assessments.

2. **IrisPredictor Integration**: Enhanced the `iris_predictor.py` to use the dosha quantification model for iris analysis.

3. **UI Implementation**: Added a dedicated "Dosha Analysis" tab in the app to display comprehensive dosha analysis results, including:
   - Dosha distribution (Vata, Pitta, Kapha percentages)
   - Metabolic health indicators
   - Organ health assessment
   - Personalized health recommendations

4. **Test Tools**:
   - `test_dosha_quantification.py` - Script to test the dosha model with sample images
   - `generate_sample_dosha_profiles.py` - Tool to generate sample dosha profiles for testing

5. **Documentation**:
   - Updated the main `README.md` with information about the dosha analysis feature
   - Created detailed documentation in `DOSHA_ANALYSIS.md`

## How to Run

### Using the Web Interface

1. Start the application with dosha analysis support:
   ```bash
   ./run_dosha_app.sh
   ```

2. Open the application in your web browser (typically at `http://localhost:8501`)

3. Navigate to the "Dosha Analysis" tab

4. Upload an iris image to perform dosha analysis

5. View your comprehensive dosha profile, metabolic indicators, organ health assessment, and personalized recommendations

### Testing the Model Directly

To test the dosha quantification model directly, use the test script:

```bash
# Test with a specific iris image
./test_dosha_quantification.py /path/to/iris/image.jpg

# Or let it find a sample image in the sample_images directory
./test_dosha_quantification.py
```

This will analyze the image and output:
- Dosha profile (Vata, Pitta, Kapha percentages)
- Metabolic indicators
- Organ health assessment
- Health recommendations

The results will also be saved as a JSON file and visualized with charts.

## Model Details

### Data Types

The model provides the following data structures:

1. **DoshaProfile**:
   - Vata, Pitta, and Kapha scores and percentages
   - Primary and secondary doshas
   - Balance score and imbalance type

2. **MetabolicIndicators**:
   - Basal metabolic rate (kcal/day)
   - Serum lipid (mg/dL)
   - Triglycerides (mg/dL)
   - CRP level (mg/L)
   - Gut diversity, enzyme activity, appetite score, and metabolism variability

3. **OrganHealthAssessment**:
   - Health score for each major organ
   - Dosha influences on the organ
   - Iris indicators related to the organ
   - Specific health recommendations

### Calculation Methodology

The model uses several analytical components:

1. **Iris Feature Extraction**: Analyzes iris color, texture, spots, and patterns to derive dosha-related features

2. **Dosha Quantification**: Maps iris features to dosha characteristics based on Ayurvedic principles

3. **Metabolic Calculation**: Derives metabolic indicators from dosha percentages and iris features

4. **Organ Assessment**: Evaluates organ health based on iris zones, dosha profile, and metabolic indicators

5. **Recommendation Generation**: Creates personalized health recommendations based on the complete analysis

## Next Steps

Potential enhancements for future development:

1. **Model Refinement**: Improve the accuracy of dosha quantification with more training data

2. **Advanced Visualization**: Add more interactive visualizations for the dosha profile and health metrics

3. **Time-Series Analysis**: Support tracking changes in dosha balance over time

4. **Integration with Wearables**: Connect dosha profiles with data from health tracking devices

5. **PDF Report Generation**: Add the ability to generate comprehensive PDF reports of the dosha analysis
