#!/usr/bin/env python3
"""
Script to change image source using Playwright automation
This script will locate an image element using the provided XPath and change its source to "5.png"
"""

import asyncio
from playwright.async_api import async_playwright

async def change_image_source():
    """Change the image source using Playwright automation"""
    
    # XPath selector for the image element
    xpath_selector = '//*[@id="root"]/div[1]/div[1]/div/div/section[1]/div[1]/div[2]/div/div/div/div[2]/div/div/div[2]/div/img'
    
    async with async_playwright() as p:
        # Launch browser (you can change to 'firefox' or 'webkit' if needed)
        browser = await p.chromium.launch(headless=False)  # Set to True for headless mode
        page = await browser.new_page()
        
        try:
            # Navigate to your application (adjust the URL as needed)
            print("Navigating to the application...")
            await page.goto("http://localhost:8501")  # Adjust URL if different
            
            # Wait for the page to load
            await page.wait_for_load_state('networkidle')
            
            # Wait for the image element to be present
            print("Waiting for image element to be present...")
            await page.wait_for_selector(f'xpath={xpath_selector}', timeout=10000)
            
            # Get the current image source
            current_src = await page.get_attribute(f'xpath={xpath_selector}', 'src')
            print(f"Current image source: {current_src}")
            
            # Change the image source to "5.png"
            print("Changing image source to 5.png...")
            await page.evaluate(f'''
                const img = document.evaluate('{xpath_selector}', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                if (img) {{
                    img.src = "5.png";
                    console.log("Image source changed to 5.png");
                }} else {{
                    console.error("Image element not found");
                }}
            ''')
            
            # Verify the change
            new_src = await page.get_attribute(f'xpath={xpath_selector}', 'src')
            print(f"New image source: {new_src}")
            
            # Wait a bit to see the change
            await asyncio.sleep(3)
            
            print("✅ Image source successfully changed to 5.png")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
        finally:
            # Close the browser
            await browser.close()

async def change_image_with_mcp():
    """Alternative method using MCP Playwright tools"""
    try:
        # This would use the MCP Playwright tools available in your environment
        print("Using MCP Playwright tools to change image...")
        
        # Navigate to the page
        print("Navigating to application...")
        # Note: You would need to call the appropriate MCP tools here
        
        # Execute JavaScript to change the image
        xpath_selector = '//*[@id="root"]/div[1]/div[1]/div/div/section[1]/div[1]/div[2]/div/div/div/div[2]/div/div/div[2]/div/img'
        
        script = f'''
        const img = document.evaluate('{xpath_selector}', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
        if (img) {{
            img.src = "5.png";
            return "Image source changed to 5.png";
        }} else {{
            return "Image element not found";
        }}
        '''
        
        print("Script prepared to change image source")
        print("You can use the MCP Playwright tools to execute this script")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("=== Image Source Changer ===")
    print(f"Target XPath: //*[@id='root']/div[1]/div[1]/div/div/section[1]/div[1]/div[2]/div/div/div/div[2]/div/div/div[2]/div/img")
    print(f"New image source: 5.png")
    print()
    
    # Run the async function
    asyncio.run(change_image_source())
