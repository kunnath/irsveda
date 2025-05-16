import traceback
from iris_zone_analyzer import IrisZoneAnalyzer

try:
    analyzer = IrisZoneAnalyzer()
    result = analyzer.process_iris_image('./irs.png')
    print('Analysis successful!')
except Exception as e:
    print(f'Error: {str(e)}')
    traceback.print_exc()
