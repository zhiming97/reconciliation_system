import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import anthropic
import json
import base64
import re
import cv2

# -------------------------------
# Configuration
# -------------------------------
# You'll need to set your Anthropic API key
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY")

# -------------------------------
# Custom CSS for Futuristic UI
# -------------------------------

def load_custom_css():
    st.markdown("""
    <style>
    /* Import futuristic font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        min-height: 100vh;
    }
    
    /* Custom title styling */
    .futuristic-title {
        font-family: 'Orbitron', monospace;
        font-weight: 900;
        font-size: 2.8rem;
        background: linear-gradient(45deg, #00d4ff, #7c3aed, #ff006e);
        background-size: 200% 200%;
        animation: gradient 3s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Subtitle styling */
    .futuristic-subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 400;
        font-size: 1.2rem;
        color: #a0a0a0;
        text-align: center;
        margin-bottom: 3rem;
        letter-spacing: 1px;
    }
    
    /* Upload container styling - removed */
    
    /* Upload section headers */
    .upload-header {
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        font-size: 1.5rem;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px dashed rgba(0, 212, 255, 0.4) !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stFileUploader > div > div:hover {
        border-color: rgba(0, 212, 255, 0.8) !important;
        background: rgba(0, 212, 255, 0.05) !important;
        transform: scale(1.02) !important;
    }
    
    .stFileUploader label {
        font-family: 'Rajdhani', sans-serif !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        font-size: 1.1rem !important;
    }
    
    /* Hide empty containers and default Streamlit elements */
    .stFileUploader > div[data-testid="stFileUploaderDropzone"]:empty {
        display: none !important;
    }
    
    div[data-testid="column"] > div:empty {
        display: none !important;
    }
    
    .element-container:empty {
        display: none !important;
    }
    
    /* Hide the default file uploader background/border */
    .stFileUploader > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Hide any residual upload container backgrounds */
    .upload-container::after {
        display: none !important;
    }
    
    /* Target specific empty upload elements */
    div[data-testid="stFileUploader"] > div:first-child:empty {
        display: none !important;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-ready {
        background-color: #00ff41;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.6);
    }
    
    .status-waiting {
        background-color: #ff9500;
        box-shadow: 0 0 10px rgba(255, 149, 0, 0.6);
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.1); }
        100% { opacity: 1; transform: scale(1); }
    }
    
    /* Image preview styling */
    .image-preview {
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        overflow: hidden;
        margin-top: 1rem;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Process button styling */
    .process-btn {
        background: linear-gradient(45deg, #00d4ff, #7c3aed) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.8rem 2.5rem !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        color: white !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 5px 20px rgba(0, 212, 255, 0.3) !important;
        margin: 2rem auto !important;
        display: block !important;
    }
    
    .process-btn:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.5) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(15, 15, 35, 0.9) !important;
        border-right: 1px solid rgba(0, 212, 255, 0.2) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        .futuristic-title {
            font-size: 2rem;
        }
        .upload-container {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# Helper Functions
# -------------------------------

def process_uploaded_file(uploaded_file):
    """Process uploaded file in memory without saving to disk"""
    # Store the original uploaded file object for direct access
    return {
        'name': uploaded_file.name,
        'size': uploaded_file.size,
        'type': uploaded_file.type,
        'file_object': uploaded_file  # Store the original file object
    }

def display_upload_status(file, file_type):
    """Display upload status with futuristic indicators"""
    if file is not None:
        st.markdown(f"""
        <div style="text-align: center; margin: 1rem 0;">
            <span class="status-indicator status-ready"></span>
            <span style="color: #00ff41; font-family: 'Rajdhani', sans-serif; font-weight: 600;">
                {file_type} UPLOADED ✓
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Display image preview
        image = Image.open(file)
        st.markdown('<div class="image-preview">', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        return True
    else:
        st.markdown(f"""
        <div style="text-align: center; margin: 1rem 0;">
            <span class="status-indicator status-waiting"></span>
            <span style="color: #ff9500; font-family: 'Rajdhani', sans-serif; font-weight: 600;">
                WAITING FOR {file_type.upper()}...
            </span>
        </div>
        """, unsafe_allow_html=True)
        return False

def create_upload_section(title, file_uploader_key, accepted_types=["png", "jpg", "jpeg"]):
    """Create a futuristic upload section"""
    st.markdown(f'<div class="upload-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="upload-header">{title}</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        f"Choose {title.lower()} file",
        type=accepted_types,
        key=file_uploader_key,
        label_visibility="collapsed"
    )
    
    file_uploaded = display_upload_status(uploaded_file, title.split()[0])
    st.markdown('</div>', unsafe_allow_html=True)
    
    return uploaded_file, file_uploaded

# -------------------------------
# Main Streamlit Application
# -------------------------------

def main():
    # Set page config
    st.set_page_config(
        page_title="Bank Reconciliation System",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Main title
    st.markdown('''
    <div class="futuristic-title">
         BANK STATEMENT RECONCILIATION SYSTEM
    </div>
    <div class="futuristic-subtitle">
        Identifying which deposits from the bank statement that are missing from SSBO
    </div>
    ''', unsafe_allow_html=True)
    
    # Create two columns for file uploads
    col1, col2 = st.columns(2, gap="large")
    
    # Initialize session state for file data
    if 'bank_statement_data' not in st.session_state:
        st.session_state.bank_statement_data = None
    if 'ssbo_deposit_data' not in st.session_state:
        st.session_state.ssbo_deposit_data = None
    
    with col1:
        bank_file, bank_uploaded = create_upload_section(
            "🏦 BANK STATEMENT", 
            "bank_statement_uploader"
        )
        
        if bank_file is not None and bank_uploaded:
            # Process the file in memory and store data in session state
            file_data = process_uploaded_file(bank_file)
            st.session_state.bank_statement_data = file_data
            st.success(f"📁 Processed: {file_data['name']} ({file_data['size']} bytes)")
            
            # Add debug button to see what Claude sees (only for bank statement)
            st.markdown("---")
            if st.button("👁️ See What Claude Sees (Bank Statement)", key="debug_btn"):
                with st.spinner("🔍 Analyzing what Claude sees..."):
                    try:
                        ocr = AnthropicOCR(ANTHROPIC_API_KEY)
                        debug_response = ocr.debug_image_extraction(
                            st.session_state.bank_statement_data['file_object']
                        )
                        
                        st.markdown("### 👁️ Raw Claude Analysis of Your Bank Statement")
                        st.markdown("**This is exactly what Claude sees when looking at your image (simulating direct paste on Claude website):**")
                        
                        # Show the raw response in a larger text area
                        st.text_area(
                            "Claude's Raw Description:", 
                            debug_response, 
                            height=400,
                            help="This is the unprocessed, raw response from Claude about what it sees in your bank statement image"
                        )
                        
                        # Show file info for context
                        st.info(f"**File being analyzed:** {bank_file.name} ({bank_file.size} bytes) - Original quality preserved")
                        
                    except Exception as e:
                        st.error(f"❌ Debug error: {str(e)}")
                        st.exception(e)
    
    with col2:
        ssbo_file, ssbo_uploaded = create_upload_section(
            "💰 SSBO DEPOSITS SCREENSHOT ONLY", 
            "ssbo_deposit_uploader"
        )
        
        if ssbo_file is not None and ssbo_uploaded:
            # Process the file in memory and store data in session state
            file_data = process_uploaded_file(ssbo_file)
            st.session_state.ssbo_deposit_data = file_data
            st.success(f"📁 Processed: {file_data['name']} ({file_data['size']} bytes)")
    
    # Process button (only show when both files are uploaded)
    if bank_uploaded and ssbo_uploaded:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Center the button only
        col_center = st.columns([1, 2, 1])
        with col_center[1]:
            process_button = st.button("🚀 INITIATE RECONCILIATION", key="process_btn")

        
        
        # Process results outside of the column structure for full width
        if process_button:
            with st.spinner("🔄 Processing images with Claude AI..."):
                try:
                    # Process bank statement with Claude
                    st.write("📊 Processing Bank Statement...")
                    bank_result = process_bank_statement_with_claude(
                        st.session_state.bank_statement_data['file_object']
                    )
                    
                    # Process SSBO deposits with Claude
                    st.write("💰 Processing SSBO Deposits...")
                    ssbo_result = process_ssbo_deposits_with_claude(
                        st.session_state.ssbo_deposit_data['file_object']
                    )
                    
                    # Display results
                    if bank_result['success'] and ssbo_result['success']:
                        st.success("✅ Both images processed successfully!")
                        
                        # Show summary in full width
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"🏦 **Bank Statement:** {bank_result['record_count']} records extracted")
                        with col2:
                            st.info(f"💰 **SSBO Deposits:** {ssbo_result['record_count']} records extracted")
                        
                        # Show raw bank data
                        with st.expander("🏦 Raw Bank Statement Data", expanded=False):
                            st.write("**Raw JSON from Claude OCR:**")
                            st.json(bank_result['data'])
                            
                            # Show as DataFrame for better readability
                            st.write("**As DataFrame:**")
                            bank_df = pd.DataFrame(bank_result['data'])
                            st.dataframe(bank_df, use_container_width=True)
                        
                        # Show raw SSBO data
                        with st.expander("💰 Raw SSBO Deposits Data", expanded=False):
                            st.write("**Raw JSON from Claude OCR:**")
                            st.json(ssbo_result['data'])
                            
                            # Show as DataFrame for better readability
                            st.write("**As DataFrame:**")
                            ssbo_df = pd.DataFrame(ssbo_result['data'])
                            st.dataframe(ssbo_df, use_container_width=True)
                        
                        # Create and display comparison table in full width
                        st.markdown("---")
                        st.markdown("## 📊 Statement Comparison")
                        
                        comparison_data = create_comparison_table(bank_result['data'], ssbo_result['data'])
                        
                        if comparison_data:
                            # Convert to DataFrame for better display
                            # Convert to DataFrame for better display
                            df = pd.DataFrame(comparison_data)
                            
                            # Apply conditional styling to the DataFrame
                            def highlight_status(val):
                                if val == "Tally":
                                    return 'background-color: #90EE90; color: #006400; font-weight: bold;'
                                elif val == "Not Tally":
                                    return 'background-color: #FFB6C1; color: #8B0000; font-weight: bold;'
                                return ''
                            
                            # Apply the styling
                            styled_df = df.style.applymap(highlight_status, subset=['Status'])
                            
                            # Display the comparison table in full width
                            st.dataframe(
                                styled_df,
                                use_container_width=True,
                                hide_index=False,
                                column_config={
                                    "Date_A": st.column_config.TextColumn("Date_A", width="medium"),
                                    "Description_A": st.column_config.TextColumn("Description_A", width="large"),
                                    "Type_A": st.column_config.TextColumn("Type_A", width="small"),
                                    "Amount_A": st.column_config.NumberColumn("Amount_A", width="medium", format="%.2f"),
                                    "Date_B": st.column_config.TextColumn("Date_B", width="medium"),
                                    "Description_B": st.column_config.TextColumn("Description_B", width="large"),
                                    "Type_B": st.column_config.TextColumn("Type_B", width="small"),
                                    "Amount_B": st.column_config.NumberColumn("Amount_B", width="medium", format="%.2f"),
                                    "Status": st.column_config.TextColumn("Status", width="small"),

                                }
                            )
                        
                        # Show summary statistics in full width
                        tally_count = len([row for row in comparison_data if row['Status'] == 'Tally'])
                        not_tally_count = len([row for row in comparison_data if row['Status'] == 'Not Tally'])
                        
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Records", len(comparison_data))
                        with col2:
                            st.metric("✅ Tally", tally_count)
                        with col3:
                            st.metric("❌ Not Tally", not_tally_count)
                        
                        # Store results in session state for further processing
                        st.session_state.bank_ocr_result = bank_result
                        st.session_state.ssbo_ocr_result = ssbo_result
                        st.session_state.comparison_data = comparison_data
                        
                    else:
                        st.error("❌ Error processing images:")
                        if not bank_result['success']:
                            st.error(f"Bank Statement: {bank_result['error']}")
                        if not ssbo_result['success']:
                            st.error(f"SSBO Deposits: {ssbo_result['error']}")
                            
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    st.exception(e)



    
class AnthropicOCR:
    def __init__(self, api_key: str):
        """
        Initialize Anthropic client for OCR operations
        
        Args:
            api_key: Your Anthropic API key
        """
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def detect_media_type_from_content(self, file_content) -> str:
        """
        Detect media type from file content using magic bytes
        """
        # Convert memoryview to bytes if needed
        if hasattr(file_content, 'tobytes'):
            file_content = file_content.tobytes()
        elif not isinstance(file_content, bytes):
            file_content = bytes(file_content)
        
        # Check file headers (magic bytes)
        if file_content.startswith(b'\xff\xd8\xff'):
            return "image/jpeg"
        elif file_content.startswith(b'\x89PNG\r\n\x1a\n'):
            return "image/png"
        elif file_content.startswith(b'GIF87a') or file_content.startswith(b'GIF89a'):
            return "image/gif"
        elif file_content.startswith(b'RIFF') and file_content[8:12] == b'WEBP':
            return "image/webp"
        else:
            # Default to PNG if we can't detect
            return "image/png"

    def encode_image_from_file(self, uploaded_file) -> str:
        """
        Encode uploaded file directly to base64 string with grayscale conversion only
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Get the file content directly from the uploaded file
            file_content = uploaded_file.getbuffer()
            
            # Convert memoryview to bytes if needed
            if hasattr(file_content, 'tobytes'):
                file_content = file_content.tobytes()
            elif not isinstance(file_content, bytes):
                file_content = bytes(file_content)
            
            # Convert to numpy array using OpenCV
            nparr = np.frombuffer(file_content, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to grayscale ONLY
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Encode the grayscale image to PNG
            _, buffer = cv2.imencode('.png', gray_image)
            base64_data = base64.b64encode(buffer).decode('utf-8')
            
            # Store the detected type for use in the API call
            uploaded_file._detected_media_type = "image/png"
            
            return base64_data
            
        except Exception as e:
            print(f"Error in encode_image_from_file: {str(e)}")
            raise e
    
 
    def extract_table_as_json(self, uploaded_file) -> str:
        """
        Extract table data from uploaded file and return as JSON string
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            JSON string of the extracted table data
        """
        
        prompt = """
        The attached image contains a structured table. Please extract ALL data from the table and return it as a JSON array of objects. 
        
        Requirements:
        1. Each row should be a JSON object
        2. Use the column headers as JSON keys
        3. Be extremely accurate with financial data.
        4. Do not hallucinate or use previous memory of other images uploaded to return the result. Always return whatever that is displayed to you in the image ONLY
        5. For event time column, convert the data into the format of YYYY-MM-DD.
        6. Only return the rows where the "Transaction Type" is "Deposit" or "Transfer".
        7. Only return the data of these 4 columns only: Transaction Type, Event Time, Amount, Remark
        
        Return ONLY the JSON array, no explanations or additional text.
        Example format:
        [
            {"column1": "value1", "column2": 123.45, "column3": "2025-08-14", "column4": "Deposit"},
            {"column1": "value2", "column2": 678.90, "column3": "2025-08-15", "column4": "Transfer"}
        ]
        """
        
        # Encode the image properly using the file object
        base64_image = self.encode_image_from_file(uploaded_file)
        
        # Use the detected media type (more reliable than file extension)
        media_type = getattr(uploaded_file, '_detected_media_type', None)
        if not media_type:
            media_type = uploaded_file.type if uploaded_file.type else "image/png"
        
        print(f"Using media type: {media_type}")
        
        # Create the message with image
        message = self.client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
        )
        
        # Extract the JSON content
        json_content = message.content[0].text
        return json_content

    def extract_bank_table_as_json(self, uploaded_file) -> str:
        """
        Extract table data from uploaded file and return as JSON string
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            JSON string of the extracted table data
        """
        
        prompt = """The attached image is a screenshot of bank transaction history. Please extract ALL data from the table and return it as a JSON array of objects. 
                
                Requirements:
                1. Each row should be a JSON object
                2. Use the column headers as JSON keys
                3. Be extremely accurate with financial data.
                4. Do not hallucinate or use previous memory of other images uploaded to return the result. Always return whatever that is displayed to you in the image ONLY
                5. For event time column, convert the data into the format of YYYY-MM-DD.
                6. The image may have colours. Please perform some pre-processing before you perform OCR to achieve best accuracy.

                From the json array, if the column such as Credit, Deposit, Money In are not null or empty, put these transactions under "Deposit" in a column called transaction type. Otherwise, put the transaction under "Transfer" in the transaction type column.
                

                Return ONLY the JSON array, no explanations or additional text.
                Example format:
                [
                    {"column1": "value1", "column2": 123.45, "column3": "2025-08-14"},
                    {"column1": "value2", "column2": 678.90, "column3": "2025-08-15"}
                ]
                """

        # Encode the image properly using the file object
        base64_image = self.encode_image_from_file(uploaded_file)
        
        # Use the detected media type (more reliable than file extension)
        media_type = getattr(uploaded_file, '_detected_media_type', None)
        if not media_type:
            media_type = uploaded_file.type if uploaded_file.type else "image/png"
        
        print(f"Using media type: {media_type}")
        
        # Create the message with image
        message = self.client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
        )
        
        # Extract the JSON content
        json_content = message.content[0].text
        return json_content
    

    
    def _clean_json_response(self, json_content: str) -> str:
        """Clean JSON response from Claude"""
        # Remove markdown code blocks if present
        json_content = re.sub(r'```json\n?', '', json_content)
        json_content = re.sub(r'```\n?', '', json_content)
        
        # Remove any leading/trailing whitespace
        json_content = json_content.strip()
        
        return json_content

    def debug_image_extraction(self, uploaded_file):
        """
        Debug helper to see what Claude is actually seeing
        Simulates pasting image directly on Claude website
        """
        prompt = """The attached image is a screenshot of bank transaction history. Please extract ALL data from the table and return it as a JSON array of objects. 
        
        Requirements:
        1. Each row should be a JSON object
        2. Use the column headers as JSON keys
        3. Be extremely accurate with financial data.
        4. Do not hallucinate or use previous memory of other images uploaded to return the result. Always return whatever that is displayed to you in the image ONLY
        5. For event time column, convert the data into the format of YYYY-MM-DD.
        6. The image may have colours. Please perform some pre-processing before you perform OCR to achieve best accuracy.        

        Return ONLY the JSON array, no explanations or additional text.
        Example format:
        [
            {"column1": "value1", "column2": 123.45, "column3": "2025-08-14"},
            {"column1": "value2", "column2": 678.90, "column3": "2025-08-15"}
        ]
        """
        
        # Encode the image directly from the uploaded file (no processing)
        base64_image = self.encode_image_from_file(uploaded_file)
        
        # Determine the correct media type from the file
        media_type = uploaded_file.type if uploaded_file.type else "image/png"
        
        # Create the message with image
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
        )
        
        # Extract the response
        response = message.content[0].text
        return response
    

# -------------------------------
# OCR Processing Functions
# -------------------------------

def create_comparison_table(bank_data, ssbo_data):
    """Create a comparison table between bank statement and SSBO deposits"""
    comparison_rows = []
    
    # Helper function to standardize date format
    def standardize_date(date_str):
        """Convert various date formats to YYYY-MM-DD"""
        try:
            # Handle different date formats
            if '/' in date_str:
                # Format: "16/8/2025" -> "2025-08-16"
                if len(date_str.split('/')) == 3:
                    parts = date_str.split('/')
                    if len(parts[2]) == 4:  # Year is 4 digits
                        day, month, year = parts
                    else:  # Year is 2 digits
                        day, month, year = parts
                        year = '20' + year
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            elif '-' in date_str:
                # Format: "2025-08-15 22:48:08" -> "2025-08-15"
                if ' ' in date_str:
                    return date_str.split(' ')[0]
                else:
                    return date_str
            return date_str
        except:
            return date_str
    
    # Normalize amount for reliable matching
    def normalize_amount(value):
        try:
            if isinstance(value, str):
                value = value.replace(',', '').strip()
            return float(value)
        except Exception:
            return value
    
    # Normalize transaction type (minimal normalization)
    def normalize_tx_type(tx):
        if tx is None:
            return None
        return str(tx).strip().lower()
    
    # Create a lookup dictionary for SSBO data by standardized date and amount
    # Use a list to track multiple transactions with same date+amount
    ssbo_lookup = {}
    for ssbo_item in ssbo_data:
        standardized_date = standardize_date(ssbo_item['Event Time'])
        normalized_amount = normalize_amount(ssbo_item['Amount'])
        normalized_type = normalize_tx_type(ssbo_item.get('Transaction Type'))
        key = (standardized_date, normalized_amount, normalized_type)
        if key not in ssbo_lookup:
            ssbo_lookup[key] = []
        ssbo_lookup[key].append(ssbo_item)
    
    # Process each bank statement entry
    for bank_item in bank_data:
        bank_date = bank_item['Event Time']
        bank_amount = normalize_amount(bank_item['Amount'])
        bank_description = bank_item['Description/Remarks']
        bank_tx_type = normalize_tx_type(bank_item.get('Transaction Type'))
        bank_tx_display = bank_item.get('Transaction Type') or (bank_tx_type.capitalize() if bank_tx_type else 'Unknown')
        
        # Standardize the bank date for comparison
        standardized_bank_date = standardize_date(bank_date)
        
        # Check if there's a matching SSBO entry using standardized dates
        ssbo_key = (standardized_bank_date, bank_amount, bank_tx_type)
        if ssbo_key in ssbo_lookup and len(ssbo_lookup[ssbo_key]) > 0:
            # Get the first available matching SSBO item
            ssbo_item = ssbo_lookup[ssbo_key].pop(0)  # Remove from available matches
            status = "Tally"
            ssbo_remark = ssbo_item['Remark']
        else:
            status = "Not Tally"
            ssbo_remark = "No matching record"
        
        comparison_rows.append({
            'Date_A': bank_date,
            'Description_A': bank_description,
            'Type_A': bank_tx_display,
            'Amount_A': bank_amount,
            'Date_B': ssbo_item['Event Time'] if status == "Tally" else "No match",
            'Description_B': ssbo_item['Remark'] if status == "Tally" else "No match",
            'Type_B': ssbo_item['Transaction Type'] if status == "Tally" else "No match",
            'Amount_B': ssbo_item['Amount'] if status == "Tally" else "No match",
            'Status': status
        })
    
    return comparison_rows

def process_bank_statement_with_claude(uploaded_file) -> dict:
    """Process bank statement image with Claude OCR"""
    try:
        ocr = AnthropicOCR(ANTHROPIC_API_KEY)
        
        # Debug info
        print(f"Processing bank statement: {uploaded_file.name}, type: {uploaded_file.type}")
        
        json_content = ocr.extract_bank_table_as_json(uploaded_file)
        json_content = ocr._clean_json_response(json_content)
        
        # Remove commas from numbers in the JSON string
        fixed_json = re.sub(r'(\d),(\d)', r'\1\2', json_content)
        
        # Parse the JSON
        json_data = json.loads(fixed_json)
        
        return {
            'success': True,
            'data': json_data,
            'record_count': len(json_data)
        }
    except Exception as e:
        print(f"Error in process_bank_statement_with_claude: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'data': None
        }

def process_ssbo_deposits_with_claude(uploaded_file) -> dict:
    """Process SSBO deposits image with Claude OCR"""
    try:
        ocr = AnthropicOCR(ANTHROPIC_API_KEY)
        
        # Debug info
        print(f"Processing SSBO deposits: {uploaded_file.name}, type: {uploaded_file.type}")
        
        json_content = ocr.extract_table_as_json(uploaded_file)
        json_content = ocr._clean_json_response(json_content)
        
        # Remove commas from numbers in the JSON string
        fixed_json = re.sub(r'(\d),(\d)', r'\1\2', json_content)
        
        # Parse the JSON
        json_data = json.loads(fixed_json)
        
        return {
            'success': True,
            'data': json_data,
            'record_count': len(json_data)
        }
    except Exception as e:
        print(f"Error in process_ssbo_deposits_with_claude: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'data': None
        }






if __name__ == '__main__':
    main()
