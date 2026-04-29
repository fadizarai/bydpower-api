import os
import re
import cv2
import random
import time
import tempfile
import pytesseract
import numpy as np
from flask import Flask, request, jsonify
from twilio.rest import Client
from inference_sdk import InferenceHTTPClient
from difflib import get_close_matches

app = Flask(__name__)

# ── Twilio Config ──
ACCOUNT_SID = 'AC74d01f126735435dc924aa317e3584f4'
AUTH_TOKEN  = '296e4a4313312e2567205f70acb0396e'
TWILIO_NUM  = '+17622473995'
twilio_client = Client(ACCOUNT_SID, AUTH_TOKEN)

# ── OTP Store ──
otp_store = {}

# ============================================
# OTP FUNCTIONS
# ============================================
def generate_otp(phone):
    code = str(random.randint(100000, 999999))
    otp_store[phone] = {
        'code': code,
        'expiry': time.time() + 300
    }
    twilio_client.messages.create(
        body=f'BYDPower - Votre code OTP: {code}. Valide 5 minutes.',
        from_=TWILIO_NUM,
        to=phone
    )
    return code

def verify_otp(phone, code):
    if phone not in otp_store:
        return False, 'Code invalide'
    entry = otp_store[phone]
    if time.time() > entry['expiry']:
        del otp_store[phone]
        return False, 'Code expire'
    if entry['code'] != code:
        return False, 'Code incorrect'
    del otp_store[phone]
    return True, 'OK'

# ============================================
# OCR FUNCTION
# ============================================
def process_carte_grise(image_path):
    img = cv2.imread(image_path)
    resultats = {}

    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="DrTVKDk0f0AMbJXDFwAr"
    )
    result = client.infer(image_path, model_id="cg-project/1")

    config_ocr = {
        'NumSerie':       {'lang': 'fra', 'psm': 7},
        'Constructor':    {'lang': 'fra', 'psm': 8},
        'TypeCommercial': {'lang': 'fra', 'psm': 6},
        'NomPrenom':      {'lang': 'ara', 'psm': 7},
    }

    def pretraiter(zone):
        zone = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
        zone = cv2.resize(zone, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        zone = cv2.fastNlMeansDenoising(zone, h=15)
        _, zone = cv2.threshold(zone, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return zone

    def nettoyer(texte, classe):
        if classe == 'NumSerie':
            t = re.sub(r'[^A-Z0-9]', '', texte.upper())
            t = t.replace('YWZ', 'VWZ').replace('WYW', 'WVW').replace('WIW', 'WVW')
            return t
        if classe == 'NomPrenom':
            return re.sub(r'[^\u0600-\u06FF\s]', '', texte).strip()
        if classe in ['Constructor', 'TypeCommercial']:
            t = re.sub(r'[^A-Z0-9\s\-]', '', texte.upper()).strip()
            return re.sub(r'^[\d\s]+', '', t).strip()
        return texte.strip()

    gouvernorats_tunisiens = [
        'تونس','صفاقس','سوسة','القيروان','بنزرت',
        'قابس','مدنين','قفصة','سيدي بوزيد','زغوان',
        'باجة','جندوبة','الكاف','سليانة','المنستير',
        'المهدية','نابل','أريانة','بن عروس','منوبة',
        'توزر','قبلي','تطاوين'
    ]

    for pred in result['predictions']:
        classe    = pred['class']
        confiance = pred['confidence']
        cx, cy = pred['x'], pred['y']
        w,  h  = pred['width'], pred['height']
        x1 = max(0,             int(cx - w/2) - 5)
        y1 = max(0,             int(cy - h/2) - 5)
        x2 = min(img.shape[1], int(cx + w/2) + 15)
        y2 = min(img.shape[0], int(cy + h/2) + 5)
        zone = img[y1:y2, x1:x2]

        if classe == 'Immatriculation':
            zone_rot = cv2.rotate(zone, cv2.ROTATE_90_CLOCKWISE)
            h_rot, w_rot = zone_rot.shape[:2]
            q = w_rot // 4
            zones_immat = {
                'numero':      zone_rot[:, 0:q],
                'gouvernorat': zone_rot[:, q:int(q*2.8)],
                'serie':       zone_rot[:, int(q*2.2):],
            }
            for nom, z in zones_immat.items():
                z = pretraiter(z)
                if nom == 'serie':
                    for col in range(z.shape[1]):
                        if z[:, col].mean() > 128:
                            z = z[:, col:]
                            break
                    r = pytesseract.image_to_string(
                        z,
                        config='--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
                    ).strip()
                    r_propre = re.sub(r'[^0-9]', '', r)
                elif nom == 'gouvernorat':
                    r = pytesseract.image_to_string(
                        z, config='--oem 3 --psm 8 -l ara'
                    ).strip()
                    r_propre = re.sub(r'[^\u0600-\u06FF]', '', r).strip()
                    if len(r_propre) >= 4:
                        r_propre = r_propre[:4]
                    r_propre = r_propre.replace('ب', 'ت')
                    if r_propre not in gouvernorats_tunisiens:
                        matches = get_close_matches(
                            r_propre, gouvernorats_tunisiens, n=1, cutoff=0.4
                        )
                        if matches:
                            r_propre = matches[0]
                else:
                    r = pytesseract.image_to_string(
                        z, config='--oem 3 --psm 8 -l fra'
                    ).strip()
                    r_propre = re.sub(r'[^0-9]', '', r)
                resultats[f'immat_{nom}'] = {
                    'valeur': r_propre, 'confiance': confiance
                }
            continue

        zone = pretraiter(zone)
        cfg = config_ocr.get(classe, {'lang': 'fra', 'psm': 7})
        texte_brut = pytesseract.image_to_string(
            zone,
            config=f'--oem 3 --psm {cfg["psm"]} -l {cfg["lang"]}'
        ).strip()
        texte_propre = nettoyer(texte_brut, classe)
        resultats[classe] = {'valeur': texte_propre, 'confiance': confiance}

    numero = resultats.get('immat_numero',      {}).get('valeur', '')
    gouv   = resultats.get('immat_gouvernorat', {}).get('valeur', '')
    serie  = resultats.get('immat_serie',       {}).get('valeur', '')

    return {
        'immatriculation': f"{numero} {gouv} {serie}".strip(),
        'serie':           resultats.get('NumSerie',       {}).get('valeur', None),
        'constructeur':    resultats.get('Constructor',    {}).get('valeur', None),
        'type_commercial': resultats.get('TypeCommercial', {}).get('valeur', None),
        'nom_prenom':      resultats.get('NomPrenom',      {}).get('valeur', None),
    }

# ============================================
# ROUTES
# ============================================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/scan', methods=['POST'])
def scan():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        result = process_carte_grise(tmp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        os.unlink(tmp_path)

@app.route('/otp/send', methods=['POST'])
def send_otp():
    data = request.get_json()
    phone = data.get('phone')
    if not phone:
        return jsonify({'error': 'Phone number required'}), 400
    try:
        generate_otp(phone)
        return jsonify({'success': True, 'message': 'OTP sent'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/otp/verify', methods=['POST'])
def verify_otp_route():
    data = request.get_json()
    phone = data.get('phone')
    code  = data.get('code')
    if not phone or not code:
        return jsonify({'error': 'Phone and code required'}), 400
    valid, message = verify_otp(phone, code)
    if valid:
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': message}), 400

# ============================================
# START
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)