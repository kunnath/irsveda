import traceback
from iris_zone_analyzer import IrisZoneAnalyzer

def test_web_app_simulation():
    """
    Simulate how the web app would use the analysis results, particularly
    to ensure no tuple assignment errors occur when manipulating the results.
    """
    try:
        print("Initializing IrisZoneAnalyzer...")
        analyzer = IrisZoneAnalyzer()
        
        # Process an iris image
        print("\nProcessing iris image...")
        results = analyzer.process_iris_image('./irs.png')
        
        if "error" in results:
            print(f"Error: {results['error']}")
            return False
        
        # Verify result types
        print("\nVerifying result types:")
        from PIL import Image
        
        # Check image types
        for key in ["original_image", "boundary_image", "zone_map"]:
            if not isinstance(results[key], Image.Image):
                print(f"Warning: {key} is not a PIL Image, it's a {type(results[key])}")
            else:
                print(f"✓ {key} is a PIL Image")
        
        # Simulate web app manipulations
        print("\nSimulating web app manipulations:")
        
        # 1. Access to zones_analysis
        print("1. Accessing zones_analysis...")
        for zone_name, zone_data in results["zones_analysis"].items():
            # Try accessing nested data like the web app would
            try:
                name = zone_data["name"]
                systems = zone_data["ayurvedic_mapping"]["systems"]
                dosha = zone_data["ayurvedic_mapping"]["dominant_dosha"]
                condition = zone_data["health_indication"]["condition"]
                print(f"  ✓ Successfully accessed data for {zone_name}")
            except Exception as e:
                print(f"  × Error accessing data for {zone_name}: {e}")
                traceback.print_exc()
                return False
        
        # 2. Access to health_summary
        print("\n2. Accessing health_summary...")
        try:
            overall_health = results["health_summary"]["overall_health"]
            dosha_balance = results["health_summary"]["dosha_balance"]
            print(f"  ✓ Overall health: {overall_health}")
            print(f"  ✓ Dosha balance: {dosha_balance}")
        except Exception as e:
            print(f"  × Error accessing health summary: {e}")
            traceback.print_exc()
            return False
            
        # Simulate modification that might happen in the web app
        print("\n3. Simulating tuple assignment that would happen in Streamlit:")
        try:
            # Create a simulated session state dict
            session_state = {}
            
            # Store results in session state
            session_state["iris_analysis_results"] = results
            
            # Try to modify values in the session state
            if "vata" in session_state["iris_analysis_results"]["health_summary"]["dosha_balance"]:
                # This is the kind of operation that might trigger tuple assignment errors
                session_state["iris_analysis_results"]["health_summary"]["dosha_balance"]["vata"] *= 100
                
            print(f"  ✓ Successfully modified values in session state")
        except Exception as e:
            print(f"  × Error modifying values in session state: {e}")
            traceback.print_exc()
            return False
            
        print("\nAll web app simulations passed! No tuple assignment errors.")
        return True
        
    except Exception as e:
        print(f"Error during web app simulation: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_web_app_simulation()
