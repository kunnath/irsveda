# PDF Generation Error Fixes

## Issues Fixed

1. **String Concatenation Error**
   - Error message: `can only concatenate str (not "int") to str`
   - Root cause: In the FPDF library's ttfonts.py file, there was an attempt to concatenate a string with an integer without converting the integer to a string first.
   - Fix: Patched the ttfonts.py file to use `str(version)` instead of directly using `version`.

2. **Font Loading Issues**
   - Issue: DejaVu fonts were causing errors when trying to load them for PDF generation.
   - Fix: Modified the IrisReportGenerator to use standard built-in fonts (helvetica) which don't require loading external TTF files.

3. **BytesIO Compatibility Issue**
   - Error message: `'_io.BytesIO' object has no attribute 'rfind'`
   - Root cause: FPDF's image function expects a filename string but we were passing BytesIO objects.
   - Fix: Modified _np_array_to_img and _fig_to_img to save images to temporary files and return the filenames.

## How to Run the Fixes

1. First, run the font issue patch:
   ```bash
   python fix_fpdf_ttfonts.py
   ```

2. The IrisReportGenerator class has been updated to:
   - Use standard fonts instead of DejaVu fonts
   - Handle images properly by saving them to temporary files

3. Test the fix with:
   ```bash
   python test_pdf_error.py
   ```

## Technical Notes

- The FPDF library expects filenames for images, not file-like objects or buffers.
- When using standard fonts in FPDF (courier, helvetica, times, symbol, zapfdingbats), you don't need to add custom fonts.
- Temporary files are created with unique names and should be cleaned up automatically by the operating system.

## Future Improvements

- Consider using a more modern PDF generation library like ReportLab which may offer better Unicode support and more features.
- If DejaVu fonts are needed for proper Unicode support, ensure the font files are correctly formatted TTF files and properly loaded.
