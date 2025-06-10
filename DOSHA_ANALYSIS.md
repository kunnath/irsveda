## Ayurvedic Dosha Analysis

The IridoVeda application now features a comprehensive Ayurvedic dosha analysis module that quantifies Vata, Pitta, and Kapha doshas from iris features and provides detailed health metrics and recommendations.

### Dosha Quantification

The dosha quantification model analyzes iris features to determine the proportions of the three fundamental doshas in Ayurvedic medicine:

- **Vata** - Associated with movement, creativity, and flexibility
- **Pitta** - Associated with transformation, metabolism, and intelligence
- **Kapha** - Associated with structure, stability, and lubrication

### Health Metrics

Based on the dosha profile, the system calculates several important health metrics:

| Metric | Description | Units |
|--------|-------------|-------|
| Basal Metabolic Rate | Energy expended at rest | kcal/day |
| Serum Lipid | Cholesterol level | mg/dL |
| Triglycerides | Fat level in blood | mg/dL |
| CRP | Inflammatory marker | mg/L |
| Gut Diversity | Microbiome health | 0-1 scale |
| Enzyme Activity | Digestive enzyme function | 0-1 scale |
| Appetite Score | Hunger and satiety | 1-10 scale |
| Metabolism Variability | Metabolic stability | 0-1 scale |

### Organ Health Assessment

The system assesses the health of major organs based on iris features and dosha influences:

- Liver
- Stomach
- Intestines
- Lungs
- Heart
- Kidneys
- Brain
- Thyroid
- Pancreas
- Lymphatic System

Each organ receives a health score (0-1) and specific recommendations based on its condition.

### Using the Dosha Analysis Feature

1. Navigate to the "Dosha Analysis" tab in the application
2. Upload a clear, well-lit image of an iris
3. The system will analyze the image and generate:
   - Dosha distribution with percentages
   - Primary and secondary dosha identification
   - Dosha balance score
   - Metabolic health indicators
   - Organ health assessment
   - Personalized health recommendations

### Testing with Sample Data

You can test the dosha analysis with sample data:

```bash
# Generate sample dosha profiles
./generate_sample_dosha_profiles.py

# Test the dosha quantification with a sample iris image
./test_dosha_quantification.py sample_images/iris1.jpg
```

### Scientific Basis

The dosha quantification model integrates traditional Ayurvedic principles with modern computational analysis techniques. The relationships between iris features and dosha characteristics are based on:

1. Traditional Ayurvedic iris interpretation (Iridology in Ayurveda)
2. Color, texture, and pattern analysis of the iris
3. Zone-based organ correlations in iridology
4. Known relationships between doshas and physiological functions

While this approach provides valuable insights, it should be considered complementary to conventional medical evaluation rather than a replacement.
