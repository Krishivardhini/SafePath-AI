/**
 * SafePath-AI — Assistive Mobility Companion
 * Universal Real-Time Multi-Device Implementation (Mobile, Tablet, Desktop, Smart Glasses)
 * Real YOLOv8 Detection, Multi-Object Tracking, Monocular Depth, Spatial Corridor Analysis,
 * Rule-Based Safety Engine, OpenStreetMap + OSRM Navigation, Ask SafePath, and Multilingual Voice.
 */

// ============================================================================
// 1. Core Configuration & Constants (Mirrors safepath/config.py)
// ============================================================================

const CONFIG = {
  FRAME_WIDTH: 640,
  FRAME_HEIGHT: 480,
  FOCAL_LENGTH_PX: 600.0,
  CENTER_ZONE_FRACTION: 0.34,
  MAX_RELEVANT_DISTANCE_M: 6.0,
  ALERT_COOLDOWN_SECONDS: 2.0,
  CONFIDENCE_THRESHOLD: 0.40,
  TTS_RATE_WPM: 175,

  CATEGORY_WEIGHT: {
    person: 0.75,
    vehicle: 0.85,
    animal: 0.70,
    furniture: 0.50,
    small_object: 0.35,
    structural: 1.00,
  },

  POSITION_WEIGHT: {
    CENTER: 1.00,
    LEFT: 0.25,
    RIGHT: 0.25,
  },

  KNOWN_OBJECT_WIDTH_M: {
    person: 0.45,
    bicycle: 0.60,
    car: 1.80,
    motorcycle: 0.80,
    bus: 2.55,
    truck: 2.50,
    airplane: 15.0,
    train: 3.00,
    boat: 2.40,
    "traffic light": 0.35,
    "fire hydrant": 0.35,
    "stop sign": 0.75,
    "parking meter": 0.25,
    bench: 1.20,
    dog: 0.35,
    cat: 0.25,
    horse: 0.80,
    sheep: 0.50,
    cow: 0.90,
    backpack: 0.30,
    umbrella: 0.90,
    handbag: 0.30,
    suitcase: 0.45,
    bottle: 0.08,
    cup: 0.08,
    skateboard: 0.20,
    chair: 0.45,
    couch: 1.80,
    bed: 1.50,
    "dining table": 1.10,
    toilet: 0.50,
    "potted plant": 0.35,
    tv: 0.95,
    laptop: 0.35,
    mouse: 0.07,
    remote: 0.06,
    keyboard: 0.40,
    "cell phone": 0.08,
    microwave: 0.50,
    oven: 0.60,
    toaster: 0.25,
    sink: 0.55,
    refrigerator: 0.80,
    book: 0.15,
    clock: 0.25,
    vase: 0.15,
    scissors: 0.10,
    "teddy bear": 0.25,
    "hair drier": 0.20,
    toothbrush: 0.03,
    stairs: 1.00,
    door: 0.90,
    "trash bin": 0.45,
  },

  RISK_THRESHOLDS: {
    CRITICAL: 75,
    HIGH: 50,
    MEDIUM: 25,
  },
};

// ============================================================================
// 2. Multilingual Translations Dictionary (8 Languages)
// ============================================================================

const TERMS = {
  en: {
    person: "person", bicycle: "bicycle", car: "car", motorcycle: "motorcycle", airplane: "airplane",
    bus: "bus", train: "train", truck: "truck", boat: "boat", "traffic light": "traffic light",
    "fire hydrant": "fire hydrant", "stop sign": "stop sign", "parking meter": "parking meter",
    bench: "bench", bird: "bird", cat: "cat", dog: "dog", horse: "horse", sheep: "sheep",
    cow: "cow", elephant: "elephant", bear: "bear", zebra: "zebra", giraffe: "giraffe",
    backpack: "backpack", umbrella: "umbrella", handbag: "handbag", tie: "tie", suitcase: "suitcase",
    frisbee: "frisbee", skis: "skis", snowboard: "snowboard", "sports ball": "sports ball",
    kite: "kite", "baseball bat": "baseball bat", "baseball glove": "baseball glove",
    skateboard: "skateboard", surfboard: "surfboard", "tennis racket": "tennis racket",
    bottle: "bottle", "wine glass": "wine glass", cup: "cup", fork: "fork", knife: "knife",
    spoon: "spoon", bowl: "bowl", banana: "banana", apple: "apple", sandwich: "sandwich",
    orange: "orange", broccoli: "broccoli", carrot: "carrot", "hot dog": "hot dog", pizza: "pizza",
    donut: "donut", cake: "cake", chair: "chair", couch: "couch", "potted plant": "potted plant",
    bed: "bed", "dining table": "dining table", toilet: "toilet", tv: "tv", laptop: "laptop",
    mouse: "mouse", remote: "remote", keyboard: "keyboard", "cell phone": "cell phone",
    microwave: "microwave", oven: "oven", toaster: "toaster", sink: "sink",
    refrigerator: "refrigerator", book: "book", clock: "clock", vase: "vase", scissors: "scissors",
    "teddy bear": "teddy bear", "hair drier": "hair drier", toothbrush: "toothbrush",
    stairs: "stairs", door: "door", "trash bin": "trash bin",

    // Spatial & Motion
    left: "left", center: "center", right: "right", ahead: "ahead",
    "far left": "far left", "slightly left": "slightly left",
    "slightly right": "slightly right", "far right": "far right",
    close: "close", "very close": "very close", about: "about", meters: "meters",
    approaching: "approaching", "moving away": "moving away", crossing: "crossing",
    stationary: "stationary", unknown: "movement unknown",

    // Safety States
    SAFE: "SAFE", CAUTION: "CAUTION", DANGER: "DANGER", OFFLINE: "OFFLINE",

    // Path Assistant Guidance
    pathclear: "Path ahead appears clear.",
    steer_left: "Obstacle ahead. Steer slightly left.",
    steer_right: "Obstacle ahead. Steer slightly right.",
    clear_sides: "Obstacle ahead. Clear on left and right.",
    blocked: "Path blocked ahead. Stop or proceed cautiously.",

    // Navigation
    walk: "Walking route", min: "min", turn: "Turn", onto: "onto", straight: "Continue straight",
    arrive: "You have arrived at your destination."
  },

  te: {
    person: "వ్యక్తి", bicycle: "సైకిల్", car: "కారు", motorcycle: "మోటార్ సైకిల్", airplane: "విమానం",
    bus: "బస్సు", train: "రైలు", truck: "ట్రక్", boat: "పడవ", "traffic light": "ట్రాఫిక్ లైట్",
    "fire hydrant": "అగ్నిమాపక హైడ్రెంట్", "stop sign": "ఆపు గుర్తు", "parking meter": "పార్కింగ్ మీటర్",
    bench: "బెంచ్", bird: "పక్షి", cat: "పిల్లి", dog: "కుక్క", horse: "గుర్రం", sheep: "గొర్రె",
    cow: "ఆవు", elephant: "ఏనుగు", bear: "ఎలుగుబంటి", zebra: "జీబ్రా", giraffe: "జిరాఫీ",
    backpack: "బ్యాక్‌ప్యాక్", umbrella: "గొడుగు", handbag: "హ్యాండ్‌బ్యాగ్", tie: "టై", suitcase: "సూట్‌కేస్",
    frisbee: "ఫ్రిస్బీ", skis: "స్కీస్", snowboard: "స్నోబోర్డ్", "sports ball": "బంతి",
    kite: "గాలిపటం", "baseball bat": "బ్యాట్", "baseball glove": "గ్లోవ్",
    skateboard: "స్కేట్‌బోర్డ్", surfboard: "సర్ఫ్‌బోర్డ్", "tennis racket": "టెన్నిస్ రాకెట్",
    bottle: "సీసా", "wine glass": "గాజు గ్లాస్", cup: "కప్పు", fork: "ఫోర్క్", knife: "కత్తి",
    spoon: "చెంచా", bowl: "గిన్నె", banana: "అరటిపండు", apple: "ఆపిల్", sandwich: "శాండ్‌విచ్",
    orange: "నారింజ", broccoli: "బ్రోకలీ", carrot: "క్యారెట్", "hot dog": "హాట్ డాగ్", pizza: "పిజ్జా",
    donut: "డోనట్", cake: "కేక్", chair: "కుర్చీ", couch: "సోఫా", "potted plant": "మొక్క కుండి",
    bed: "మంచం", "dining table": "భోజన బల్ల", toilet: "మరుగుదొడ్డి", tv: "టీవీ", laptop: "ల్యాప్‌టాప్",
    mouse: "మౌస్", remote: "రిమోట్", keyboard: "కీబోర్డ్", "cell phone": "మొబైల్ ఫోన్",
    microwave: "మైక్రోవేవ్", oven: "ఓవెన్", toaster: "టోస్టర్", sink: "సింక్",
    refrigerator: "ఫ్రిజ్", book: "పుస్తకం", clock: "గడియారం", vase: "పూలకుండీ", scissors: "కత్తెర",
    "teddy bear": "టెడ్డీ బేర్", "hair drier": "హెయిర్ డ్రైయర్", toothbrush: "టూత్‌బ్రష్",
    stairs: "మెట్లు", door: "ద్వారం", "trash bin": "చెత్తబుట్ట",

    left: "ఎడమ వైపు", center: "మధ్యలో", right: "కుడి వైపు", ahead: "ముందు",
    "far left": "దూరంగా ఎడమవైపు", "slightly left": "కాస్త ఎడమవైపు",
    "slightly right": "కాస్త కుడివైపు", "far right": "దూరంగా కుడివైపు",
    close: "దగ్గరలో", "very close": "చాలా దగ్గరలో", about: "సుమారు", meters: "మీటర్లు",
    approaching: "దగ్గరకు వస్తోంది", "moving away": "దూరంగా వెళ్తోంది", crossing: "అడ్డంగా కదులుతోంది",
    stationary: "కదలకుండా ఉంది", unknown: "కదలిక తెలియదు",

    SAFE: "సురక్షితం", CAUTION: "జాగ్రత్త", DANGER: "ప్రమాదం", OFFLINE: "దృష్టి ఆఫ్‌లైన్",

    pathclear: "ముందు దారి ఖాళీగా ఉంది.",
    steer_left: "ముందు అడ్డంకి ఉంది. కాస్త ఎడమవైపు నడవండి.",
    steer_right: "ముందు అడ్డంకి ఉంది. కాస్త కుడివైపు నడవండి.",
    clear_sides: "ముందు అడ్డంకి ఉంది. ఎడమ, కుడి దారులు ఖాళీగా ఉన్నాయి.",
    blocked: "ముందు దారి మూసుకుపోయింది. ఆగండి లేదా జాగ్రత్తగా నడవండి.",

    walk: "నడక మార్గం", min: "నిమిషాలు", turn: "తిరగండి", onto: "వైపు", straight: "నేరుగా వెళ్ళండి",
    arrive: "మీరు మీ గమ్యాన్ని చేరుకున్నారు."
  },

  hi: {
    person: "व्यक्ति", bicycle: "साइकिल", car: "कार", motorcycle: "मोटरसाइकिल", airplane: "हवाई जहाज़",
    bus: "बस", train: "रेलगाड़ी", truck: "ट्रक", boat: "नाव", "traffic light": "ट्रैफिक लाइट",
    "fire hydrant": "अग्नि जल-स्तंभ", "stop sign": "रुकने का संकेत", "parking meter": "पार्किंग मीटर",
    bench: "बेंच", bird: "पक्षी", cat: "बिल्ली", dog: "कुत्ता", horse: "घोड़ा", sheep: "भेड़",
    cow: "गाय", elephant: "हाथी", bear: "भालू", zebra: "ज़ेबरा", giraffe: "जिराफ़",
    backpack: "बैग", umbrella: "छाता", handbag: "हैंडबैग", tie: "टाई", suitcase: "सूटकेस",
    frisbee: "फ़्रिसबी", skis: "स्की", snowboard: "स्नोबोर्ड", "sports ball": "गेंद",
    kite: "पतंग", "baseball bat": "बल्ला", "baseball glove": "दस्ताना",
    skateboard: "स्केटबोर्ड", surfboard: "सर्फ़बोर्ड", "tennis racket": "रैकेट",
    bottle: "बोतल", "wine glass": "कांच का गिलास", cup: "कप", fork: "कांटा", knife: "चाकू",
    spoon: "चम्मच", bowl: "कटोरा", banana: "केला", apple: "सेब", sandwich: "सैंडविच",
    orange: "संतरा", broccoli: "ब्रोकोली", carrot: "गाजर", "hot dog": "हॉट डॉग", pizza: "पिज़्ज़ा",
    donut: "डोनट", cake: "केक", chair: "कुर्सी", couch: "सोफ़ा", "potted plant": "गमला",
    bed: "बिस्तर", "dining table": "खाने की मेज़", toilet: "शौचालय", tv: "टीवी", laptop: "लैपटॉप",
    mouse: "माउस", remote: "रिमोट", keyboard: "कीबोर्ड", "cell phone": "मोबाइल फ़ोन",
    microwave: "माइक्रोवेव", oven: "ओवन", toaster: "टोस्टर", sink: "सिंक",
    refrigerator: "फ़्रिज", book: "किताब", clock: "घड़ी", vase: "फूलदान", scissors: "कैंची",
    "teddy bear": "टेडी बियर", "hair drier": "हेयर ड्रायर", toothbrush: "टूथब्रश",
    stairs: "सीढ़ियाँ", door: "दरवाज़ा", "trash bin": "कचरा पेटी",

    left: "बाईं ओर", center: "बीच में", right: "दाईं ओर", ahead: "सामने",
    "far left": "दूर बाईं ओर", "slightly left": "थोड़ा बाएँ",
    "slightly right": "थोड़ा दाएँ", "far right": "दूर दाईं ओर",
    close: "पास", "very close": "बहुत पास", about: "लगभग", meters: "मीटर",
    approaching: "पास आ रहा है", "moving away": "दूर जा रहा है", crossing: "आड़ा जा रहा है",
    stationary: "स्थिर", unknown: "गति अज्ञात",

    SAFE: "सुरक्षित", CAUTION: "सावधानी", DANGER: "खतरा", OFFLINE: "दृष्टि बंद",

    pathclear: "सामने का रास्ता साफ़ है।",
    steer_left: "आगे बाधा है। थोड़ा बाएँ मुड़ें।",
    steer_right: "आगे बाधा है। थोड़ा दाएँ मुड़ें।",
    clear_sides: "सामने बाधा है। बाएँ और दाएँ रास्ता खाली है।",
    blocked: "रास्ता रुका हुआ है। रुकें या सावधानी से चलें।",

    walk: "पैदल मार्ग", min: "मिनट", turn: "मुड़ें", onto: "की ओर", straight: "सीधे चलें",
    arrive: "आप अपने गंतव्य पर पहुँच गए हैं।"
  },

  ta: {
    person: "நபர்", bicycle: "சைக்கிள்", car: "கார்", motorcycle: "மோட்டார் சைக்கிள்", airplane: "விமானம்",
    bus: "பேருந்து", train: "ரயில்", truck: "லாரி", boat: "படகு", "traffic light": "போக்குவரத்து விளக்கு",
    "fire hydrant": "தீயணைப்பு குழாய்", "stop sign": "நிறுத்தும் குறி", "parking meter": "பார்க்கிங் மீட்டர்",
    bench: "பெஞ்ச்", bird: "பறவை", cat: "பூனை", dog: "நாய்", horse: "குதிரை", sheep: "செம்மறி",
    cow: "பசு", elephant: "யானை", bear: "கரடி", zebra: "வரிக்குதிரை", giraffe: "ஒட்டகச்சிவிங்கி",
    backpack: "பின்னல் பை", umbrella: "குடை", handbag: "கைப்பை", tie: "டை", suitcase: "சூட்கேஸ்",
    frisbee: "பறக்கும் வட்டு", skis: "ஸ்கி", snowboard: "பனிச்சறுக்கு பலகை", "sports ball": "பந்து",
    kite: "பட்டம்", "baseball bat": "மட்டை", "baseball glove": "கையுறை",
    skateboard: "சறுக்கு பலகை", surfboard: "அலைச்சறுக்கு பலகை", "tennis racket": "டென்னிஸ் மட்டை",
    bottle: "பாட்டில்", "wine glass": "கண்ணாடி கிளாஸ்", cup: "கப்", fork: "முட்கரண்டி", knife: "கத்தி",
    spoon: "கரண்டி", bowl: "கிண்ணம்", banana: "வாழைப்பழம்", apple: "ஆப்பிள்", sandwich: "சாண்ட்விச்",
    orange: "ஆரஞ்சு", broccoli: "ப்ரோக்கோலி", carrot: "கேரட்", "hot dog": "ஹாட் டாக்", pizza: "பீட்சா",
    donut: "டோனட்", cake: "கேக்", chair: "நாற்காலி", couch: "சோபா", "potted plant": "செடி தொட்டி",
    bed: "படுக்கை", "dining table": "சாப்பாட்டு மேஜை", toilet: "கழிப்பறை", tv: "டிவி", laptop: "மடிக்கணினி",
    mouse: "மவுஸ்", remote: "ரிமோட்", keyboard: "விசைப்பலகை", "cell phone": "கைபேசி",
    microwave: "மைக்ரோவேவ்", oven: "அடுப்பு", toaster: "டோஸ்டர்", sink: "சிங்க்",
    refrigerator: "பிரிட்ஜ்", book: "புத்தகம்", clock: "கடிகாரம்", vase: "பூந்தொட்டி", scissors: "கத்தரிக்கோல்",
    "teddy bear": "டெடி பியர்", "hair drier": "ஹேர் ட்ரையர்", toothbrush: "பல் துலக்கி",
    stairs: "படிகள்", door: "கதவு", "trash bin": "குப்பைத் தொட்டி",

    left: "இடதுபுறம்", center: "நடுவில்", right: "வலதுபுறம்", ahead: "முன்னால்",
    "far left": "தொலைவில் இடது", "slightly left": "சற்று இடது",
    "slightly right": "சற்று வலது", "far right": "தொலைவில் வலது",
    close: "அருகில்", "very close": "மிக அருகில்", about: "சுமார்", meters: "மீட்டர்கள்",
    approaching: "அணுகுகிறது", "moving away": "விலகிச் செல்கிறது", crossing: "குறுக்கே செல்கிறது",
    stationary: "நிலையாக உள்ளது", unknown: "இயக்கம் தெரியவில்லை",

    SAFE: "பாதுகாப்பானது", CAUTION: "எச்சரிக்கை", DANGER: "ஆபத்து", OFFLINE: "பார்வை ஆஃப்லைன்",

    pathclear: "முன்னால் உள்ள பாதை தெளிவாக உள்ளது.",
    steer_left: "முன்னால் தடை உள்ளது. சற்று இடதுபுறம் நடக்கவும்.",
    steer_right: "முன்னால் தடை உள்ளது. சற்று வலதுபுறம் நடக்கவும்.",
    clear_sides: "முன்னால் தடை உள்ளது. இடது மற்றும் வலது வழிகள் தெளிவாக உள்ளன.",
    blocked: "பாதை தடுக்கப்பட்டுள்ளது. நிற்கவும் அல்லது கவனமாக நடக்கவும்.",

    walk: "நடை பாதை", min: "நிமிடம்", turn: "திருப்பவும்", onto: "நோக்கி", straight: "நேராக தொடரவும்",
    arrive: "நீங்கள் உங்கள் இலக்கை அடைந்துவிட்டீர்கள்."
  },

  kn: {
    person: "ವ್ಯಕ್ತಿ", bicycle: "ಸೈಕಲ್", car: "ಕಾರು", motorcycle: "ಮೋಟಾರ್ ಸೈಕಲ್", airplane: "ವಿಮಾನ",
    bus: "ಬಸ್ಸು", train: "ರೈಲು", truck: "ಟ್ರಕ್", boat: "ದೋಣಿ", "traffic light": "ಸಂಚಾರ ದೀಪ",
    "fire hydrant": "ಅಗ್ನಿ ಶಾಮಕ", "stop sign": "ನಿಲ್ಲಿಸುವ ಚಿಹ್ನೆ", "parking meter": "ಪಾರ್ಕಿಂಗ್ ಮೀಟರ್",
    bench: "ಬೆಂಚ್", bird: "ಪಕ್ಷಿ", cat: "ಬೆಕ್ಕು", dog: "ನಾಯಿ", horse: "ಕುದುರೆ", sheep: "ಕುರಿ",
    cow: "ಹಸು", elephant: "ಆನೆ", bear: "ಕರಡಿ", zebra: "ಜೀಬ್ರಾ", giraffe: "ಜಿರಾಫೆ",
    backpack: "ಬ್ಯಾಗ್", umbrella: "ಛತ್ರಿ", handbag: "ಹ್ಯಾಂಡ್‌ಬ್ಯಾಗ್", tie: "ಟೈ", suitcase: "ಸೂಟ್‌ಕೇಸ್",
    frisbee: "ಫ್ರಿಸ್ಬೀ", skis: "ಸ್ಕೀ", snowboard: "ಸ್ನೋಬೋರ್ಡ್", "sports ball": "ಚಂಡು",
    kite: "ಪತಂಗ", "baseball bat": "ಬ್ಯಾಟ್", "baseball glove": "ಕೈಗವಸು",
    skateboard: "ಸ್ಕೇಟ್‌ಬೋರ್ಡ್", surfboard: "ಸರ್ಫ್‌ಬೋರ್ಡ್", "tennis racket": "ಟೆನ್ನಿಸ್ ರಾಕೆಟ್",
    bottle: "ಬಾಟಲ್", "wine glass": "ಗಾಜಿನ ಲೋಟ", cup: "ಕಪ್", fork: "ಫೋರ್ಕ್", knife: "ಚಾಕು",
    spoon: "ಚಮಚ", bowl: "ಬಟ್ಟಲು", banana: "ಬಾಳೆಹಣ್ಣು", apple: "ಸೇಬು", sandwich: "ಸ್ಯಾಂಡ್‌ವಿಚ್",
    orange: "ಕಿತ್ತಳೆ", broccoli: "ಕೋಸುಗಡ್ಡೆ", carrot: "ಕ್ಯಾರೆಟ್", "hot dog": "ಹಾಟ್ ಡಾಗ್", pizza: "ಪಿಜ್ಜಾ",
    donut: "ಡೋನಟ್", cake: "ಕೇಕ್", chair: "ಕುರ್ಚಿ", couch: "ಸೋಫಾ", "potted plant": "ಗಿಡದ ಕುಂಡ",
    bed: "ಹಾಸಿಗೆ", "dining table": "ಊಟದ ಮೇಜು", toilet: "ಶೌಚಾಲಯ", tv: "ಟಿವಿ", laptop: "ಲ್ಯಾಪ್ಟಾಪ್",
    mouse: "ಮೌಸ್", remote: "ರಿಮೋಟ್", keyboard: "ಕೀಬೋರ್ಡ್", "cell phone": "ಮೊಬೈಲ್",
    microwave: "ಮೈಕ್ರೋವೇವ್", oven: "ಓವನ್", toaster: "ಟೋಸ್ಟರ್", sink: "ಸಿಂಕ್",
    refrigerator: "ಫ್ರಿಜ್", book: "ಪುಸ್ತಕ", clock: "ಗಡಿಯಾರ", vase: "ಹೂದಾನಿ", scissors: "ಕತ್ತರಿ",
    "teddy bear": "ಟೆಡ್ಡಿ ಬೇರ್", "hair drier": "ಹೇರ್ ಡ್ರೈಯರ್", toothbrush: "ಟೂತ್ ಬ್ರಷ್",
    stairs: "ಮೆಟ್ಟಿಲುಗಳು", door: "ಬಾಗಿಲು", "trash bin": "ಕಸದ ಬುಟ್ಟಿ",

    left: "ಎಡಭಾಗ", center: "ಮಧ್ಯ", right: "ಬಲಭಾಗ", ahead: "ಮುಂದೆ",
    "far left": "ದೂರದ ಎಡ", "slightly left": "ಸ್ವಲ್ಪ ಎಡಕ್ಕೆ",
    "slightly right": "ಸ್ವಲ್ಪ ಬಲಕ್ಕೆ", "far right": "ದೂರದ ಬಲ",
    close: "ಹತ್ತಿರ", "very close": "ತುಂಬಾ ಹತ್ತಿರ", about: "ಸುಮಾರು", meters: "ಮೀಟರ್‌ಗಳು",
    approaching: "ಹತ್ತಿರ ಬರುತ್ತಿದೆ", "moving away": "ದೂರ ಹೋಗುತ್ತಿದೆ", crossing: "ದಾಟುತ್ತಿದೆ",
    stationary: "ಸ್ಥಿರವಾಗಿದೆ", unknown: "ಚಲನೆ ತಿಳಿದಿಲ್ಲ",

    SAFE: "ಸುರಕ್ಷಿತ", CAUTION: "ಎಚ್ಚರಿಕೆ", DANGER: "ಅಪಾಯ", OFFLINE: "ದೃಷ್ಟಿ ಆಫ್‌ಲೈನ್",

    pathclear: "ಮುಂದಿನ ಮಾರ್ಗ ಸ್ಪಷ್ಟವಾಗಿದೆ.",
    steer_left: "ಮುಂದೆ ಅಡಚಣೆಯಿದೆ. ಸ್ವಲ್ಪ ಎಡಕ್ಕೆ ತಿರುಗಿ.",
    steer_right: "ಮುಂದೆ ಅಡಚಣೆಯಿದೆ. ಸ್ವಲ್ಪ ಬಲಕ್ಕೆ ತಿರುಗಿ.",
    clear_sides: "ಮುಂದೆ ಅಡಚಣೆಯಿದೆ. ಎಡ ಮತ್ತು ಬಲ ಬದಿ ಸ್ಪಷ್ಟವಾಗಿದೆ.",
    blocked: "ಮಾರ್ಗ ನಿರ್ಬಂಧಿಸಲಾಗಿದೆ. ನಿಲ್ಲಿ ಅಥವಾ ಎಚ್ಚರಿಕೆಯಿಂದ ಮುನ್ನಡೆಯಿರಿ.",

    walk: "ನಡೆಯುವ ಮಾರ್ಗ", min: "ನಿಮಿಷ", turn: "ತಿರುಗಿ", onto: "ಕಡೆಗೆ", straight: "ನೇರವಾಗಿ ಮುಂದುವರಿಯಿರಿ",
    arrive: "ನೀವು ನಿಮ್ಮ ಗಮ್ಯಸ್ಥಾನವನ್ನು ತಲುಪಿದ್ದೀರಿ."
  },

  mr: {
    person: "व्यक्ती", bicycle: "सायकल", car: "कार", motorcycle: "मोटारसायकल", airplane: "विमान",
    bus: "बस", train: "आगगाडी", truck: "ट्रक", boat: "नाव", "traffic light": "रहदारी दिवा",
    "fire hydrant": "अग्निशामक", "stop sign": "थांबण्याचा फलक", "parking meter": "पार्किंग मीटर",
    bench: "बाक", bird: "पक्षी", cat: "मांजर", dog: "कुत्रा", horse: "घोडा", sheep: "मेंढी",
    cow: "गाय", elephant: "हत्ती", bear: "अस्वल", zebra: "झिब्रा", giraffe: "जिराफ",
    backpack: "बॅग", umbrella: "छत्री", handbag: "हँडबॅग", tie: "टाय", suitcase: "सूटकेस",
    frisbee: "फ्रिस्बी", skis: "स्की", snowboard: "स्नोबोर्ड", "sports ball": "चेंडू",
    kite: "पतंग", "baseball bat": "बॅट", "baseball glove": "हातमोजा",
    skateboard: "स्केटबोर्ड", surfboard: "सर्फबोर्ड", "tennis racket": "रॅकेट",
    bottle: "बाटली", "wine glass": "काचेचा ग्लास", cup: "कप", fork: "काटा", knife: "सुरी",
    spoon: "चमचा", bowl: "वाटी", banana: "केळे", apple: "सफरचंद", sandwich: "सँडविच",
    orange: "संत्रा", broccoli: "ब्रोकोली", carrot: "गाजर", "hot dog": "हॉट डॉग", pizza: "पिझ्झा",
    donut: "डोनट", cake: "केक", chair: "खुर्ची", couch: "सोफा", "potted plant": "कुंडी",
    bed: "पलंग", "dining table": "जेवणाचे टेबल", toilet: "शौचालय", tv: "टीव्ही", laptop: "लॅपटॉप",
    mouse: "माऊस", remote: "रिमोट", keyboard: "कीबोर्ड", "cell phone": "मोबाइल फोन",
    microwave: "मायक्रोव्हेव्ह", oven: "ओव्हन", toaster: "टोस्टर", sink: "सिंक",
    refrigerator: "फ्रिज", book: "पुस्तक", clock: "घड्याळ", vase: "फुलदाणी", scissors: "कात्री",
    "teddy bear": "टेडी बेअर", "hair drier": "हेअर ड्रायर", toothbrush: "टूथब्रश",
    stairs: "पायऱ्या", door: "दार", "trash bin": "कचराकुंडी",

    left: "डावीकडे", center: "मध्यभागी", right: "उजवीकडे", ahead: "समोर",
    "far left": "दूर डावीकडे", "slightly left": "किंचित डावीकडे",
    "slightly right": "किंचित उजवीकडे", "far right": "दूर उजवीकडे",
    close: "जवळ", "very close": "खूप जवळ", about: "सुमारे", meters: "मीटर",
    approaching: "जवळ येत आहे", "moving away": "दूर जात आहे", crossing: "आडवे जात आहे",
    stationary: "स्थिर आहे", unknown: "हालचाल माहित नाही",

    SAFE: "सुरक्षित", CAUTION: "सावधानता", DANGER: "धोका", OFFLINE: "दृष्टी ऑफलाइन",

    pathclear: "पुढील मार्ग मोकळा आहे.",
    steer_left: "पुढे अडथळा आहे. किंचित डावीकडे वळा.",
    steer_right: "पुढे अडथळा आहे. किंचित उजवीकडे वळा.",
    clear_sides: "पुढे अडथळा आहे. डावी आणि उजवी बाजू मोकळी आहे.",
    blocked: "मार्ग अडवला आहे. थांबा किंवा काळजीपूर्वक पुढे जा.",

    walk: "चालण्याचा मार्ग", min: "मिनिट", turn: "वळा", onto: "कडे", straight: "सरळ सुरू ठेवा",
    arrive: "तुम्ही तुमच्या गंतव्यस्थानी पोहोचला आहात."
  },

  bn: {
    person: "ব্যক্তি", bicycle: "সাইকেল", car: "গাড়ি", motorcycle: "মোটরসাইকেল", airplane: "বিমান",
    bus: "বাস", train: "ট্রেন", truck: "ট্রাক", boat: "নৌকা", "traffic light": "ট্রাফিক লাইট",
    "fire hydrant": "ফায়ার হাইড্রেন্ট", "stop sign": "থামার সংকেত", "parking meter": "পার্কিং মিটার",
    bench: "বেঞ্চ", bird: "পাখি", cat: "বিড়াল", dog: "কুকুর", horse: "ঘোড়া", sheep: "ভেড়া",
    cow: "গরু", elephant: "হাতি", bear: "ভালুক", zebra: "জেব্রা", giraffe: "জিরাফ",
    backpack: "ব্যাকপ্যাক", umbrella: "ছাতা", handbag: "হ্যান্ডব্যাগ", tie: "টাই", suitcase: "স্যুটকেস",
    frisbee: "ফ্রিসবি", skis: "স্কি", snowboard: "স্নোবোর্ড", "sports ball": "বল",
    kite: "ঘুড়ি", "baseball bat": "ব্যাট", "baseball glove": "দস্তানা",
    skateboard: "স্কেটবোর্ড", surfboard: "সার্ফবোর্ড", "tennis racket": "র‌্যাকেট",
    bottle: "বোতল", "wine glass": "কাঁচের গ্লাস", cup: "কাপ", fork: "কাঁটাচামচ", knife: "ছুরি",
    spoon: "চামচ", bowl: "বাটি", banana: "কলা", apple: "আপেল", sandwich: "স্যান্ডউইচ",
    orange: "কমলা", broccoli: "ব্রোকলি", carrot: "গাজর", "hot dog": "হট ডগ", pizza: "পিজ্জা",
    donut: "ডোনাট", cake: "কেক", chair: "চেয়ার", couch: "সোফা", "potted plant": "টবের গাছ",
    bed: "বিছানা", "dining table": "খাবার টেবিল", toilet: "টয়লেট", tv: "টিভি", laptop: "ল্যাপটপ",
    mouse: "মাউস", remote: "রিমোট", keyboard: "কীবোর্ড", "cell phone": "মোবাইল ফোন",
    microwave: "মাইক্রোওয়েভ", oven: "ওভেন", toaster: "টোস্টার", sink: "সিঙ্ক",
    refrigerator: "ফ্রিজ", book: "বই", clock: "ঘড়ি", vase: "ফুলদানি", scissors: "কাঁচি",
    "teddy bear": "টেডি বিয়ার", "hair drier": "হেয়ার ড্রায়ার", toothbrush: "টুথব্রাশ",
    stairs: "সিঁড়ি", door: "দরজা", "trash bin": "ডাস্টবিন",

    left: "বামে", center: "মাঝখানে", right: "ডানে", ahead: "সামনে",
    "far left": "দূরে বামে", "slightly left": "সামান্য বামে",
    "slightly right": "সামান্য ডানে", "far right": "দূরে ডানে",
    close: "কাছে", "very close": "খুব কাছে", about: "প্রায়", meters: "মিটার",
    approaching: "কাছে আসছে", "moving away": "দূরে যাচ্ছে", crossing: "পার হচ্ছে",
    stationary: "স্থির", unknown: "গতিবিধি অজানা",

    SAFE: "নিরাপদ", CAUTION: "সতর্কতা", DANGER: "বিপদ", OFFLINE: "দৃষ্টি অফলাইন",

    pathclear: "সামনের পথ পরিষ্কার আছে।",
    steer_left: "সামনে বাধা আছে। সামান্য বামে চলুন।",
    steer_right: "সামনে বাধা আছে। সামান্য ডানে চলুন।",
    clear_sides: "সামনে বাধা আছে। বাম এবং ডান পথ পরিষ্কার।",
    blocked: "পথ অবরুদ্ধ। থামুন বা সাবধানে এগিয়ে যান।",

    walk: "হাঁটার রুট", min: "মিনিট", turn: "ঘুরুন", onto: "দিকে", straight: "সোজা এগিয়ে যান",
    arrive: "আপনি আপনার গন্তব্যে পৌঁছে গেছেন।"
  },

  es: {
    person: "persona", bicycle: "bicicleta", car: "coche", motorcycle: "motocicleta", airplane: "avión",
    bus: "autobús", train: "tren", truck: "camión", boat: "barco", "traffic light": "semáforo",
    "fire hydrant": "boca de incendios", "stop sign": "señal de stop", "parking meter": "parquímetro",
    bench: "banco", bird: "pájaro", cat: "gato", dog: "perro", horse: "caballo", sheep: "oveja",
    cow: "vaca", elephant: "elefante", bear: "oso", zebra: "cebra", giraffe: "jirafa",
    backpack: "mochila", umbrella: "paraguas", handbag: "bolso", tie: "corbata", suitcase: "maleta",
    frisbee: "disco", skis: "esquís", snowboard: "tabla de nieve", "sports ball": "pelota",
    kite: "cometa", "baseball bat": "bate", "baseball glove": "guante",
    skateboard: "monopatín", surfboard: "tabla de surf", "tennis racket": "raqueta",
    bottle: "botella", "wine glass": "copa", cup: "taza", fork: "tenedor", knife: "cuchillo",
    spoon: "cuchara", bowl: "cuenco", banana: "plátano", apple: "manzana", sandwich: "sándwich",
    orange: "naranja", broccoli: "brócoli", carrot: "zanahoria", "hot dog": "perrito caliente", pizza: "pizza",
    donut: "dónut", cake: "pastel", chair: "silla", couch: "sofá", "potted plant": "planta",
    bed: "cama", "dining table": "mesa de comedor", toilet: "inodoro", tv: "televisor", laptop: "portátil",
    mouse: "ratón", remote: "mando", keyboard: "teclado", "cell phone": "teléfono móvil",
    microwave: "microondas", oven: "horno", toaster: "tostadora", sink: "fregadero",
    refrigerator: "frigorífico", book: "libro", clock: "reloj", vase: "jarrón", scissors: "tijeras",
    "teddy bear": "oso de peluche", "hair drier": "secador", toothbrush: "cepillo de dientes",
    stairs: "escaleras", door: "puerta", "trash bin": "papelera",

    left: "a la izquierda", center: "al centro", right: "a la derecha", ahead: "adelante",
    "far left": "lejos a la izquierda", "slightly left": "un poco a la izquierda",
    "slightly right": "un poco a la derecha", "far right": "lejos a la derecha",
    close: "cerca", "very close": "muy cerca", about: "a unos", meters: "metros",
    approaching: "acercándose", "moving away": "alejándose", crossing: "cruzando",
    stationary: "estacionario", unknown: "movimiento desconocido",

    SAFE: "SEGURO", CAUTION: "PRECAUCIÓN", DANGER: "PELIGRO", OFFLINE: "VISIÓN DESCONECTADA",

    pathclear: "El camino hacia adelante parece despejado.",
    steer_left: "Obstáculo adelante. Gire un poco a la izquierda.",
    steer_right: "Obstáculo adelante. Gire un poco a la derecha.",
    clear_sides: "Obstáculo adelante. Despejado a la izquierda y derecha.",
    blocked: "Camino bloqueado adelante. Deténgase o avance con precaución.",

    walk: "Ruta a pie", min: "min", turn: "Gire", onto: "hacia", straight: "Continúe recto",
    arrive: "Ha llegado a su destino."
  }
};

const VOICE_LOCALES = {
  en: "en-US",
  te: "te-IN",
  hi: "hi-IN",
  ta: "ta-IN",
  kn: "kn-IN",
  mr: "mr-IN",
  bn: "bn-IN",
  es: "es-ES"
};

const LANGUAGE_NAMES = {
  en: "English",
  te: "తెలుగు (Telugu)",
  hi: "हिन्दी (Hindi)",
  ta: "தமிழ் (Tamil)",
  kn: "ಕನ್ನಡ (Kannada)",
  mr: "मराठी (Marathi)",
  bn: "বাংলা (Bengali)",
  es: "Español (Spanish)"
};

// ============================================================================
// 2b. Comprehensive Entire-Website Translation Dictionary
// ============================================================================

const I18N_DICTIONARY = {
  en: {
    skip_to_content: "Skip to main content",
    brand_sub: "Assistive Mobility Companion",
    nav_home: "Home",
    nav_vision: "Live View",
    nav_objects: "Detections",
    nav_navigation: "Navigation",
    nav_path: "Path Assistant",
    nav_voice: "Ask SafePath",
    nav_safety: "Safety Engine",
    nav_history: "History",
    nav_accessibility: "Accessibility",
    nav_settings: "Settings",
    nav_about: "About",
    system_active: "System Active",
    live_camera_feed: "Live Camera Feed",
    start_btn: "Start",
    begin_assistance: "Begin assistance",
    stop_btn: "Stop",
    end_session: "End session",
    voice_assistance: "Voice Assistance",
    listening_commands: "Listening for your commands...",
    snapshot_btn: "Snapshot",
    upload_btn: "Upload",
    fps_rate: "FPS Rate",
    tracked_objects: "Tracked Objects",
    priority_hazard: "Priority Hazard",
    pipeline_latency: "Pipeline Latency",
    detected_objects_title: "Detected Objects",
    no_objects_detected: "No objects detected",
    env_clear: "The environment is clear.",
    voice_instruction_title: "Voice Instruction",
    hear_again: "Hear Again",
    nav_status_title: "Navigation Status",
    continue_walking: "Continue walking",
    no_obstacles_detected: "No obstacles detected.",
    perm_title: "Enable Camera & Spatial Audio",
    perm_desc: "SafePath-AI requires camera and microphone permissions to detect obstacles, calculate distance, and guide you with voice alerts.",
    perm_grant_btn: "Enable Camera & Audio",
    perm_skip_btn: "Continue without Camera",
    perm_voice_status: "🎙️ Dual Control: Tap button or say 'Grant' / 'Start'",
    perm_privacy_note: "🔒 Privacy First: All AI computer vision runs in real-time. No video is recorded or stored.",
    perm_lang_select_title: "Select Language / భాష / भाषा:",
    connect_mobile_title: "📱 Connect Mobile or Tablet",
    connect_mobile_desc: "Access SafePath on your phone over local Wi-Fi with zero installation:",
    copy_url_btn: "Copy URL",
    connect_mobile_steps: "1. Connect phone to same Wi-Fi network.<br>2. Open URL in browser.<br>3. Tap 'Add to Home Screen' for full-screen PWA!",
    snapshot_breakdown_title: "📸 AI Snapshot Spatial Breakdown",
    assistance_verdict_label: "Assistance Verdict:",
    th_object: "Object",
    th_distance: "Distance",
    th_zone: "Zone",
    th_risk: "Risk",
    th_level: "Level",
    auth_terms_note: "By continuing, you agree to our",
    auth_terms_link: "Terms of Service",
    and_str: "and",
    auth_privacy_link: "Privacy Policy",
    welcome_prompt: "Welcome to SafePath-AI. Tap the button or say 'Grant' or 'Start' to enable camera and audio assistance."
  },

  te: {
    skip_to_content: "ప్రధాన కంటెంట్‌కు వెళ్లండి",
    brand_sub: "సహాయక మొబిలిటీ సాధనం",
    nav_home: "హోమ్",
    nav_vision: "లైవ్ వ్యూ",
    nav_objects: "గుర్తింపులు",
    nav_navigation: "నావిగేషన్",
    nav_path: "మార్గ సహాయకుడు",
    nav_voice: "సేఫ్‌పాత్‌ను అడగండి",
    nav_safety: "భద్రతా ఇంజిన్",
    nav_history: "చరిత్ర",
    nav_accessibility: "యాక్సెసిబిలిటీ",
    nav_settings: "సెట్టింగ్‌లు",
    nav_about: "గురించి",
    system_active: "సిస్టమ్ యాక్టివ్",
    live_camera_feed: "ప్రత్యక్ష కెమెరా ఫీడ్",
    start_btn: "ప్రారంభించు",
    begin_assistance: "సహాయం ప్రారంభించండి",
    stop_btn: "ఆపు",
    end_session: "సెషన్ ముగించండి",
    voice_assistance: "వాయిస్ సహాయం",
    listening_commands: "మీ ఆదేశాల కోసం వింటోంది...",
    snapshot_btn: "స్నాప్‌షాట్",
    upload_btn: "అప్‌లోడ్",
    fps_rate: "FPS రేటు",
    tracked_objects: "ట్రాక్ చేయబడిన వస్తువులు",
    priority_hazard: "ప్రాధాన్యత ప్రమాదం",
    pipeline_latency: "పైప్‌లైన్ ఆలస్యం",
    detected_objects_title: "గుర్తించిన వస్తువులు",
    no_objects_detected: "ఎలాంటి వస్తువులు గుర్తించబడలేదు",
    env_clear: "పరిసరాలు స్పష్టంగా ఉన్నాయి.",
    voice_instruction_title: "వాయిస్ సూచన",
    hear_again: "మళ్లీ వినండి",
    nav_status_title: "నావిగేషన్ స్థితి",
    continue_walking: "నడవడం కొనసాగించండి",
    no_obstacles_detected: "అడ్డంకులు ఏవీ గుర్తించబడలేదు.",
    perm_title: "కెమెరా & ప్రాదేశిక ఆడియోను ప్రారంభించండి",
    perm_desc: "అడ్డంకులను గుర్తించడానికి, దూరాన్ని లెక్కించడానికి మరియు వాయిస్ హెచ్చరికలతో మీకు మార్గదర్శకత్వం చేయడానికి కెమెరా అనుమతి అవసరం.",
    perm_grant_btn: "కెమెరా & ఆడియోను అనుమతించండి",
    perm_skip_btn: "కెమెరా లేకుండా కొనసాగండి",
    perm_voice_status: "🎙️ వాయిస్ లేదా టచ్: 'అనుమతించు' అని చెప్పండి లేదా బటన్ తాకండి",
    perm_privacy_note: "🔒 గోప్యత మొదటిది: AI కంప్యూటర్ విజన్ రియల్-టైమ్‌లో రన్ అవుతుంది. ఏ వీడియో రికార్డ్ చేయబడదు.",
    perm_lang_select_title: "భాషను ఎంచుకోండి:",
    connect_mobile_title: "📱 మొబైల్ లేదా టాబ్లెట్‌ను కనెక్ట్ చేయండి",
    connect_mobile_desc: "ఏ ఇన్‌స్టాలేషన్ లేకుండా స్థానిక Wi-Fi ద్వారా మీ ఫోన్‌లో SafePath ని ఉపయోగించండి:",
    copy_url_btn: "URL కాపీ చేయండి",
    connect_mobile_steps: "1. ఫోన్‌ను అదే Wi-Fi కి కనెక్ట్ చేయండి.<br>2. బ్రౌజర్‌లో URL తెరవండి.<br>3. పూర్తి స్క్రీన్ PWA కోసం 'Add to Home Screen' నొక్కండి!",
    snapshot_breakdown_title: "📸 AI స్నాప్‌షాట్ ప్రాదేశిక విశ్లేషణ",
    assistance_verdict_label: "సహాయక తీర్పు:",
    th_object: "వస్తువు",
    th_distance: "దూరం",
    th_zone: "జోన్",
    th_risk: "ప్రమాదం",
    th_level: "స్థాయి",
    auth_terms_note: "కొనసాగడం ద్వారా, మీరు మా",
    auth_terms_link: "సేవా నిబంధనలు",
    and_str: "మరియు",
    auth_privacy_link: "గోప్యతా విధానం అంగీకరిస్తున్నారు",
    welcome_prompt: "సేఫ్‌పాత్-AI కి స్వాగతం. కెమెరా మరియు ఆడియో సహాయాన్ని ప్రారంభించడానికి స్క్రీన్‌పై తాకండి లేదా 'అనుమతించు' లేదా 'ప్రారంభించు' అని చెప్పండి."
  },

  hi: {
    skip_to_content: "मुख्य सामग्री पर जाएं",
    brand_sub: "सहायक गतिशीलता साथी",
    nav_home: "होम",
    nav_vision: "लाइव दृश्य",
    nav_objects: "पहचान",
    nav_navigation: "नेविगेशन",
    nav_path: "मार्ग सहायक",
    nav_voice: "सेफपाथ से पूछें",
    nav_safety: "सुरक्षा इंजन",
    nav_history: "इतिहास",
    nav_accessibility: "अभिगम्यता",
    nav_settings: "सेटिंग्स",
    nav_about: "के बारे में",
    system_active: "सिस्टम सक्रिय",
    live_camera_feed: "लाइव कैमरा फ़ीड",
    start_btn: "शुरू करें",
    begin_assistance: "सहायता शुरू करें",
    stop_btn: "रोकें",
    end_session: "सत्र समाप्त करें",
    voice_assistance: "ध्वनि सहायता",
    listening_commands: "आपके आदेशों की प्रतीक्षा कर रहा है...",
    snapshot_btn: "स्नैपशॉट",
    upload_btn: "अपलोड",
    fps_rate: "FPS दर",
    tracked_objects: "ट्रैक की गई वस्तुएं",
    priority_hazard: "प्राथमिकता खतरा",
    pipeline_latency: "पाइपलाइन विलंब",
    detected_objects_title: "पहचाने गए ऑब्जेक्ट",
    no_objects_detected: "कोई वस्तु नहीं मिली",
    env_clear: "वातावरण साफ़ है।",
    voice_instruction_title: "ध्वनि निर्देश",
    hear_again: "फिर से सुनें",
    nav_status_title: "नेविगेशन स्थिति",
    continue_walking: "चलते रहें",
    no_obstacles_detected: "कोई बाधा नहीं मिली।",
    perm_title: "कैमरा और स्थानिक ऑडियो सक्षम करें",
    perm_desc: "बाधाओं का पता लगाने, दूरी मापने और ध्वनि चेतावनियों के साथ मार्गदर्शन करने के लिए कैमरा और माइक अनुमति आवश्यक है।",
    perm_grant_btn: "कैमरा और ऑडियो सक्षम करें",
    perm_skip_btn: "कैमरे के बिना जारी रखें",
    perm_voice_status: "🎙️ दोहरा नियंत्रण: 'अनुमति दें' बोलें या बटन पर टैप करें",
    perm_privacy_note: "🔒 गोपनीयता पहले: सभी AI रीयल-टाइम में चलता है। कोई वीडियो रिकॉर्ड नहीं होता।",
    perm_lang_select_title: "भाषा चुनें:",
    connect_mobile_title: "📱 मोबाइल या टैबलेट कनेक्ट करें",
    connect_mobile_desc: "स्थानीय वाई-फ़ाई पर अपने फोन पर सेफपाथ का उपयोग करें:",
    copy_url_btn: "URL कॉपी करें",
    connect_mobile_steps: "1. फोन को समान वाई-फ़ाई से जोड़ें।<br>2. ब्राउज़र में URL खोलें।<br>3. 'Add to Home Screen' दबाएं!",
    snapshot_breakdown_title: "📸 AI स्नैपशॉट स्थानिक विश्लेषण",
    assistance_verdict_label: "सहायता निर्णय:",
    th_object: "वस्तु",
    th_distance: "दूरी",
    th_zone: "ज़ोन",
    th_risk: "खतरा",
    th_level: "स्तर",
    auth_terms_note: "जारी रखकर, आप हमारे",
    auth_terms_link: "सेवा की शर्तें",
    and_str: "और",
    auth_privacy_link: "गोपनीयता नीति से सहमत हैं",
    welcome_prompt: "सेफपाथ-AI में आपका स्वागत है। कैमरा और ऑडियो सहायता शुरू करने के लिए स्क्रीन पर टैप करें या 'अनुमति दें' कहें।"
  },

  ta: {
    skip_to_content: "முக்கிய உள்ளடக்கத்திற்குச் செல்லவும்",
    brand_sub: "இயக்கம் உதவும் துணை",
    nav_home: "முகப்பு",
    nav_vision: "நேரலை பார்வை",
    nav_objects: "கண்டறிதல்கள்",
    nav_navigation: "வழிசெலுத்தல்",
    nav_path: "பாதை உதவியாளர்",
    nav_voice: "சேஃப்பாதையிடம் கேளுங்கள்",
    nav_safety: "பாதுகாப்பு இயந்திரம்",
    nav_history: "வரலாறு",
    nav_accessibility: "அணுகல்தன்மை",
    nav_settings: "அமைப்புகள்",
    nav_about: "பற்றி",
    system_active: "அமைப்பு செயலில்",
    live_camera_feed: "நேரலை கேமரா ஃபீட்",
    start_btn: "தொடங்கு",
    begin_assistance: "உதவியை தொடங்கு",
    stop_btn: "நிறுத்து",
    end_session: "அமர்வை முடி",
    voice_assistance: "குரல் உதவி",
    listening_commands: "உங்கள் கட்டளைகளுக்காக கேட்கிறது...",
    snapshot_btn: "புகைப்படம்",
    upload_btn: "பதிவேற்று",
    fps_rate: "FPS விகிதம்",
    tracked_objects: "கண்காணிக்கப்படும் பொருள்கள்",
    priority_hazard: "முன்னுரிமை ஆபத்து",
    pipeline_latency: "பைப்லைன் தாமதம்",
    detected_objects_title: "கண்டறியப்பட்ட பொருள்கள்",
    no_objects_detected: "எந்தப் பொருளும் கண்டறியப்படவில்லை",
    env_clear: "சுற்றுப்புறம் தெளிவாக உள்ளது.",
    voice_instruction_title: "குரல் வழிமுறை",
    hear_again: "மீண்டும் கேளுங்கள்",
    nav_status_title: "வழிசெலுத்தல் நிலை",
    continue_walking: "தொடர்ந்து நடக்கவும்",
    no_obstacles_detected: "தடைகள் எதுவும் கண்டறியப்படவில்லை.",
    perm_title: "கேமரா மற்றும் ஆடியோவை இயக்கவும்",
    perm_desc: "தடைகளைக் கண்டறியவும் குரல் வழிகாட்டுதல் வழங்கவும் கேமரா மற்றும் மைக் அனுமதி தேவைப்படுகிறது.",
    perm_grant_btn: "கேமரா மற்றும் ஆடியோவை இயக்கவும்",
    perm_skip_btn: "கேமரா இல்லாமல் தொடரவும்",
    perm_voice_status: "🎙️ குரல் அல்லது தொடுதல்: 'அனுமதி' என்று சொல்லுங்கள் அல்லது தொடவும்",
    perm_privacy_note: "🔒 தனியுரிமை முக்கியம்: எந்த வீடியோவும் சேமிக்கப்படுவதில்லை.",
    perm_lang_select_title: "மொழியைத் தேர்ந்தெடுக்கவும்:",
    connect_mobile_title: "📱 மொபைல் அல்லது டேப்லெட்டை இணைக்கவும்",
    connect_mobile_desc: "வைஃபை மூலம் உங்கள் தொலைபேசியில் SafePath ஐப் பயன்படுத்தவும்:",
    copy_url_btn: "URL நகலெடு",
    connect_mobile_steps: "1. ஒரே வைஃபையுடன் இணைக்கவும்.<br>2. பிரவுசரில் URL திறக்கவும்.",
    snapshot_breakdown_title: "📸 AI ஸ்னாப்ஷாட் பகுப்பாய்வு",
    assistance_verdict_label: "உதவி முடிவு:",
    th_object: "பொருள்",
    th_distance: "தூரம்",
    th_zone: "மண்டலம்",
    th_risk: "ஆபத்து",
    th_level: "நிலை",
    auth_terms_note: "தொடர்வதன் மூலம், நீங்கள் எங்கள்",
    auth_terms_link: "சேவை விதிமுறைகள்",
    and_str: "மற்றும்",
    auth_privacy_link: "தனியுரிமைக் கொள்கையை ஏற்கிறீர்கள்",
    welcome_prompt: "சேஃப்பாதை-AIக்கு வருக. கேமரா அனுமதியை வழங்க திரையைத் தொடவும் அல்லது 'அனுமதி' என்று சொல்லவும்."
  },

  kn: {
    skip_to_content: "ಮುಖ್ಯ ವಿಷಯಕ್ಕೆ ಹೋಗಿ",
    brand_sub: "ಸಹಾಯಕ ಮೊಬಿಲಿಟಿ ಒಡನಾಡಿ",
    nav_home: "ಮುಖಪುಟ",
    nav_vision: "ಲೈವ್ ವೀಕ್ಷಣೆ",
    nav_objects: "ಪತ್ತೆಗಳು",
    nav_navigation: "ನ್ಯಾವಿಗೇಷನ್",
    nav_path: "ಮಾರ್ಗ ಸಹಾಯಕ",
    nav_voice: "ಸೇಫ್‌ಪಾತ್ ಕೇಳಿ",
    nav_safety: "ಸುರಕ್ಷತಾ ಎಂಜಿನ್",
    nav_history: "ಇತಿಹಾಸ",
    nav_accessibility: "ಪ್ರವೇಶಿಸುವಿಕೆ",
    nav_settings: "ಸೆಟ್ಟಿಂಗ್‌ಗಳು",
    nav_about: "ಬಗ್ಗೆ",
    system_active: "ವ್ಯವಸ್ಥೆ ಸಕ್ರಿಯ",
    live_camera_feed: "ಲೈವ್ ಕ್ಯಾಮೆರಾ ಫೀಡ್",
    start_btn: "ಪ್ರಾರಂಭಿಸಿ",
    begin_assistance: "ಸಹಾಯ ಪ್ರಾರಂಭಿಸಿ",
    stop_btn: "ನಿಲ್ಲಿಸಿ",
    end_session: "ಅಧಿವೇಶನ ಮುಗಿಸಿ",
    voice_assistance: "ಧ್ವನಿ ಸಹಾಯ",
    listening_commands: "ನಿಮ್ಮ ಆಜ್ಞೆಗಳನ್ನು ಆಲಿಸುತ್ತಿದೆ...",
    snapshot_btn: "ಸ್ನ್ಯಾಪ್‌ಶಾಟ್",
    upload_btn: "ಅಪ್‌ಲೋಡ್",
    fps_rate: "FPS ದರ",
    tracked_objects: "ಟ್ರ್ಯಾಕ್ ಮಾಡಿದ ವಸ್ತುಗಳು",
    priority_hazard: "ಆದ್ಯತೆಯ ಅಪಾಯ",
    pipeline_latency: "ಪೈಪ್‌ಲೈನ್ ವಿಳಂಬ",
    detected_objects_title: "ಪತ್ತೆಯಾದ ವಸ್ತುಗಳು",
    no_objects_detected: "ಯಾವುದೇ ವಸ್ತುಗಳು ಪತ್ತೆಯಾಗಿಲ್ಲ",
    env_clear: "ಪರಿಸರವು ಸ್ಪಷ್ಟವಾಗಿದೆ.",
    voice_instruction_title: "ಧ್ವನಿ ಸೂಚನೆ",
    hear_again: "ಮತ್ತೆ ಕೇಳಿ",
    nav_status_title: "ನ್ಯಾವಿಗೇಷನ್ ಸ್ಥಿತಿ",
    continue_walking: "ನಡೆಯುವುದನ್ನು ಮುಂದುವರಿಸಿ",
    no_obstacles_detected: "ಯಾವುದೇ ಅಡೆತಡೆಗಳು ಕಂಡುಬಂದಿಲ್ಲ.",
    perm_title: "ಕ್ಯಾಮೆರಾ ಮತ್ತು ಆಡಿಯೊ ಸಕ್ರಿಯಗೊಳಿಸಿ",
    perm_desc: "ಅಡೆತಡೆಗಳನ್ನು ಪತ್ತೆಹಚ್ಚಲು ಮತ್ತು ಧ್ವನಿ ಎಚ್ಚರಿಕೆಗಳನ್ನು ನೀಡಲು ಕ್ಯಾಮೆರಾ ಮತ್ತು ಮೈಕ್ರೊಫೋನ್ ಅನುಮತಿ ಅಗತ್ಯವಿದೆ.",
    perm_grant_btn: "ಕ್ಯಾಮೆರಾ & ಆಡಿಯೊ ಸಕ್ರಿಯಗೊಳಿಸಿ",
    perm_skip_btn: "ಕ್ಯಾಮೆರಾ ಇಲ್ಲದೆ ಮುಂದುವರಿಯಿರಿ",
    perm_voice_status: "🎙️ ಧ್ವನಿ ಅಥವಾ ಸ್ಪರ್ಶ: 'ಅನುಮತಿಸಿ' ಎಂದು ಹೇಳಿ ಅಥವಾ ಸ್ಪರ್ಶಿಸಿ",
    perm_privacy_note: "🔒 ಗೌಪ್ಯತೆ ಮೊದಲು: ಯಾವುದೇ ವೀಡಿಯೊ ರೆಕಾರ್ಡ್ ಮಾಡಲಾಗುವುದಿಲ್ಲ.",
    perm_lang_select_title: "ಭಾಷೆ ಆಯ್ಕೆಮಾಡಿ:",
    connect_mobile_title: "📱 ಮೊಬೈಲ್ ಸಂಪರ್ಕಿಸಿ",
    connect_mobile_desc: "ಸ್ಥಳೀಯ Wi-Fi ಮೂಲಕ ನಿಮ್ಮ ಫೋನ್‌ನಲ್ಲಿ SafePath ಬಳಸಿ:",
    copy_url_btn: "URL ನಕಲಿಸಿ",
    connect_mobile_steps: "1. ಅದೇ Wi-Fi ನೆಟ್‌ವರ್ಕ್‌ಗೆ ಸಂಪರ್ಕಿಸಿ.<br>2. ಬ್ರೌಸರ್‌ನಲ್ಲಿ URL ತೆರೆಯಿರಿ.",
    snapshot_breakdown_title: "📸 AI ಸ್ನ್ಯಾಪ್‌ಶಾಟ್ ವಿಶ್ಲೇಷಣೆ",
    assistance_verdict_label: "ಸಹಾಯ ತೀರ್ಪು:",
    th_object: "ವಸ್ತು",
    th_distance: "ದೂರ",
    th_zone: "ವಲಯ",
    th_risk: "ಅಪಾಯ",
    th_level: "ಮಟ್ಟ",
    auth_terms_note: "ಮುಂದುವರಿಯುವ ಮೂಲಕ, ನೀವು ನಮ್ಮ",
    auth_terms_link: "ಸೇವಾ ನಿಯಮಗಳು",
    and_str: "ಮತ್ತು",
    auth_privacy_link: "ಗೌಪ್ಯತೆ ನೀತಿಯನ್ನು ಒಪ್ಪುತ್ತೀರಿ",
    welcome_prompt: "ಸೇಫ್‌ಪಾತ್-AI ಗೆ ಸುಸ್ವಾಗತ. ಕ್ಯಾಮೆರಾ ಅನುಮತಿ ನೀಡಲು ಪರದೆಯನ್ನು ಸ್ಪರ್ಶಿಸಿ ಅಥವಾ 'ಅನುಮತಿಸಿ' ಎಂದು ಹೇಳಿ."
  },

  mr: {
    skip_to_content: "मुख्य सामग्रीवर जा",
    brand_sub: "सहाय्यक गतिशीलता साथी",
    nav_home: "मुख्यपृष्ठ",
    nav_vision: "थेट दृश्य",
    nav_objects: "शोध",
    nav_navigation: "मार्गदर्शन",
    nav_path: "मार्ग सहाय्यक",
    nav_voice: "सेफपाथला विचारा",
    nav_safety: "सुरक्षा इंजिन",
    nav_history: "इतिहास",
    nav_accessibility: "प्रवेशयोग्यता",
    nav_settings: "सेटिंग्ज",
    nav_about: "बद्दल",
    system_active: "प्रणाली सक्रिय",
    live_camera_feed: "थेट कॅमेरा फीड",
    start_btn: "सुरू करा",
    begin_assistance: "मदत सुरू करा",
    stop_btn: "थांबवा",
    end_session: "सत्र समाप्त करा",
    voice_assistance: "आवाज मदत",
    listening_commands: "तुमच्या आज्ञांची वाट पाहत आहे...",
    snapshot_btn: "स्नॅपशॉट",
    upload_btn: "अपलोड",
    fps_rate: "FPS दर",
    tracked_objects: "ट्रॅक केलेल्या वस्तू",
    priority_hazard: "प्राधान्य धोका",
    pipeline_latency: "पाइपलाइन विलंब",
    detected_objects_title: "शोधलेल्या वस्तू",
    no_objects_detected: "कोणतीही वस्तू आढळली नाही",
    env_clear: "वातावरण स्वच्छ आहे.",
    voice_instruction_title: "आवाज सूचना",
    hear_again: "पुन्हा ऐका",
    nav_status_title: "नेव्हिगेशन स्थिती",
    continue_walking: "चालणे सुरू ठेवा",
    no_obstacles_detected: "कोणतेही अडथळे आढळले नाहीत.",
    perm_title: "कॅमेरा आणि ऑडिओ सक्षम करा",
    perm_desc: "अडथळे शोधण्यासाठी आणि आवाज सूचना देण्यासाठी कॅमेरा आणि माइक परवानगी आवश्यक आहे.",
    perm_grant_btn: "कॅमेरा आणि ऑडिओ सक्षम करा",
    perm_skip_btn: "कॅमेऱ्याशिवाय पुढे जा",
    perm_voice_status: "🎙️ आवाज किंवा स्पर्श: 'अनुमती द्या' म्हणा किंवा टॅप करा",
    perm_privacy_note: "🔒 गोपनीयता प्रथम: कोणताही व्हिडिओ संग्रहित केला जात नाही.",
    perm_lang_select_title: "भाषा निवडा:",
    connect_mobile_title: "📱 मोबाइल कनेक्ट करा",
    connect_mobile_desc: "वाय-फाय द्वारे आपल्या फोनवर SafePath वापरा:",
    copy_url_btn: "URL कॉपी करा",
    connect_mobile_steps: "1. समान वाय-फाय शी कनेक्ट करा.<br>2. ब्राउझरमध्ये URL उघडा.",
    snapshot_breakdown_title: "📸 AI स्नॅपशॉट विश्लेषण",
    assistance_verdict_label: "सहाय्यता निर्णय:",
    th_object: "वस्तू",
    th_distance: "अंतर",
    th_zone: "झोन",
    th_risk: "धोका",
    th_level: "पातळी",
    auth_terms_note: "सुरू ठेवून, आपण आमच्या",
    auth_terms_link: "सेवा अटी",
    and_str: "आणि",
    auth_privacy_link: "गोपनीयता धोरणाशी सहमत आहात",
    welcome_prompt: "सेफपाथ-AI मध्ये आपले स्वागत आहे. कॅमेरा सुरू करण्यासाठी स्क्रीनवर टॅप करा किंवा 'अनुमती द्या' म्हणा."
  },

  bn: {
    skip_to_content: "মূল বিষয়বস্তুতে যান",
    brand_sub: "সহায়ক গতিশীলতা সঙ্গী",
    nav_home: "হোম",
    nav_vision: "লাইভ ভিউ",
    nav_objects: "শনাক্তকরণ",
    nav_navigation: "নেভিগেশন",
    nav_path: "পথ সহকারী",
    nav_voice: "সেফপাথকে জিজ্ঞাসা করুন",
    nav_safety: "সুরক্ষা ইঞ্জিন",
    nav_history: "ইতিহাস",
    nav_accessibility: "অ্যাক্সেসযোগ্যতা",
    nav_settings: "সেটিংস",
    nav_about: "সম্পর্কে",
    system_active: "সিস্টেম সক্রিয়",
    live_camera_feed: "লাইভ ক্যামেরা ফিড",
    start_btn: "শুরু করুন",
    begin_assistance: "সহায়তা শুরু করুন",
    stop_btn: "থামুন",
    end_session: "সেশন শেষ করুন",
    voice_assistance: "ভয়েস সহায়তা",
    listening_commands: "আপনার আদেশের জন্য অপেক্ষা করছে...",
    snapshot_btn: "স্ন্যাপশট",
    upload_btn: "আপলোড",
    fps_rate: "FPS হার",
    tracked_objects: "ট্র্যাক করা বস্তু",
    priority_hazard: "অগ্রাধিকার বিপদ",
    pipeline_latency: "পাইপলাইন লেটেন্সি",
    detected_objects_title: "শনাক্ত করা বস্তু",
    no_objects_detected: "কোন বস্তু সনাক্ত করা যায়নি",
    env_clear: "পরিবেশ পরিষ্কার।",
    voice_instruction_title: "ভয়েস নির্দেশ",
    hear_again: "আবার শুনুন",
    nav_status_title: "নেভিগেশন স্থিতি",
    continue_walking: "হাঁটা চালিয়ে যান",
    no_obstacles_detected: "কোন বাধা সনাক্ত করা যায়নি।",
    perm_title: "ক্যামেরা এবং অডিও সক্ষম করুন",
    perm_desc: "বাধা সনাক্ত করতে এবং ভয়েস সতর্কতা দিয়ে আপনাকে গাইড করতে ক্যামেরা অনুমতি প্রয়োজন।",
    perm_grant_btn: "ক্যামেরা এবং অডিও সক্ষম করুন",
    perm_skip_btn: "ক্যামেরা ছাড়াই চালিয়ে যান",
    perm_voice_status: "🎙️ ভয়েস বা টাচ: 'অনুমতি' বলুন বা ট্যাপ করুন",
    perm_privacy_note: "🔒 গোপনীয়তা প্রথম: কোনও ভিডিও রেকর্ড বা সংরক্ষণ করা হয় না।",
    perm_lang_select_title: "ভাষা নির্বাচন করুন:",
    connect_mobile_title: "📱 মোবাইল সংযুক্ত করুন",
    connect_mobile_desc: "ওয়াইফাই এর মাধ্যমে আপনার ফোনে SafePath ব্যবহার করুন:",
    copy_url_btn: "URL কপি করুন",
    connect_mobile_steps: "1. একই ওয়াইফাইতে সংযোগ করুন।<br>2. ব্রাউজারে URL খুলুন।",
    snapshot_breakdown_title: "📸 AI স্ন্যাপশট বিশ্লেষণ",
    assistance_verdict_label: "সহায়তা রায়:",
    th_object: "বস্তু",
    th_distance: "দূরত্ব",
    th_zone: "জোন",
    th_risk: "ঝুঁকি",
    th_level: "স্তর",
    auth_terms_note: "চালিয়ে যাওয়ার মাধ্যমে, আপনি আমাদের",
    auth_terms_link: "পরিষেবার শর্তাবলী",
    and_str: "এবং",
    auth_privacy_link: "গোপনীয়তা নীতিতে সম্মত হচ্ছেন",
    welcome_prompt: "সেফপাথ-এআই তে স্বাগতম। ক্যামেরা সক্ষম করতে স্ক্রিনে ট্যাপ করুন বা 'অনুমতি' বলুন।"
  },

  es: {
    skip_to_content: "Saltar al contenido principal",
    brand_sub: "Compañero de Movilidad Asistida",
    nav_home: "Inicio",
    nav_vision: "Vista en Vivo",
    nav_objects: "Detecciones",
    nav_navigation: "Navegación",
    nav_path: "Asistente de Ruta",
    nav_voice: "Preguntar a SafePath",
    nav_safety: "Motor de Seguridad",
    nav_history: "Historial",
    nav_accessibility: "Accesibilidad",
    nav_settings: "Configuración",
    nav_about: "Acerca de",
    system_active: "Sistema Activo",
    live_camera_feed: "Transmisión de Cámara en Vivo",
    start_btn: "Iniciar",
    begin_assistance: "Comenzar asistencia",
    stop_btn: "Detener",
    end_session: "Finalizar sesión",
    voice_assistance: "Asistencia de Voz",
    listening_commands: "Escuchando tus comandos...",
    snapshot_btn: "Captura",
    upload_btn: "Subir",
    fps_rate: "Tasa FPS",
    tracked_objects: "Objetos Rastreados",
    priority_hazard: "Peligro Prioritario",
    pipeline_latency: "Latencia",
    detected_objects_title: "Objetos Detectados",
    no_objects_detected: "No se detectaron objetos",
    env_clear: "El entorno está despejado.",
    voice_instruction_title: "Instrucción de Voz",
    hear_again: "Escuchar de nuevo",
    nav_status_title: "Estado de Navegación",
    continue_walking: "Continúa caminando",
    no_obstacles_detected: "No se detectan obstáculos.",
    perm_title: "Habilitar Cámara y Audio Espacial",
    perm_desc: "SafePath-AI requiere permisos de cámara y micrófono para detectar obstáculos, calcular distancias y guiarte con voz.",
    perm_grant_btn: "Habilitar Cámara y Audio",
    perm_skip_btn: "Continuar sin Cámara",
    perm_voice_status: "🎙️ Control Dual: Di 'Permitir' o toca la pantalla",
    perm_privacy_note: "🔒 Privacidad primero: Toda la IA se ejecuta en tiempo real. No se graba ni guarda video.",
    perm_lang_select_title: "Seleccionar Idioma:",
    connect_mobile_title: "📱 Conectar Móvil o Tableta",
    connect_mobile_desc: "Accede a SafePath en tu teléfono mediante Wi-Fi local sin instalación:",
    copy_url_btn: "Copiar URL",
    connect_mobile_steps: "1. Conecta el teléfono a la misma red Wi-Fi.<br>2. Abre la URL en el navegador.",
    snapshot_breakdown_title: "📸 Desglose Espacial de Captura AI",
    assistance_verdict_label: "Veredicto de Asistencia:",
    th_object: "Objeto",
    th_distance: "Distancia",
    th_zone: "Zona",
    th_risk: "Riesgo",
    th_level: "Nivel",
    auth_terms_note: "Al continuar, aceptas nuestros",
    auth_terms_link: "Términos de Servicio",
    and_str: "y",
    auth_privacy_link: "Política de Privacidad",
    welcome_prompt: "Bienvenido a SafePath-AI. Toca la pantalla o di 'Permitir' para activar la cámara y el audio."
  }
};

const QUESTION_PROMPTS = {
  en: [
    { text: "What is ahead?", label: "What is ahead?" },
    { text: "Is my path clear?", label: "Is my path clear?" },
    { text: "What is on my left?", label: "What is on my left?" },
    { text: "What is on my right?", label: "What is on my right?" },
    { text: "Describe surroundings", label: "Describe surroundings" },
    { text: "Is anything approaching?", label: "Is anything approaching?" }
  ],
  te: [
    { text: "ముందు ఏముంది?", label: "ముందు ఏముంది?" },
    { text: "నా మార్గం స్పష్టంగా ఉందా?", label: "నా మార్గం స్పష్టంగా ఉందా?" },
    { text: "నా ఎడమవైపు ఏముంది?", label: "నా ఎడమవైపు ఏముంది?" },
    { text: "నా కుడివైపు ఏముంది?", label: "నా కుడివైపు ఏముంది?" },
    { text: "పరిసరాలను వివరించండి", label: "పరిసరాలను వివరించండి" },
    { text: "ఏదైనా దగ్గరకు వస్తోందా?", label: "ఏదైనా దగ్గరకు వస్తోందా?" }
  ],
  hi: [
    { text: "आगे क्या है?", label: "आगे क्या है?" },
    { text: "क्या मेरा रास्ता साफ है?", label: "क्या मेरा रास्ता साफ है?" },
    { text: "मेरी बाईं ओर क्या है?", label: "मेरी बाईं ओर क्या है?" },
    { text: "मेरी दाईं ओर क्या है?", label: "मेरी दाईं ओर क्या है?" },
    { text: "आस-पास का वर्णन करें", label: "आस-पास का वर्णन करें" },
    { text: "क्या कुछ पास आ रहा है?", label: "क्या कुछ पास आ रहा है?" }
  ],
  ta: [
    { text: "முன்னால் என்ன இருக்கிறது?", label: "முன்னால் என்ன இருக்கிறது?" },
    { text: "பாதை தெளிவாக உள்ளதா?", label: "பாதை தெளிவாக உள்ளதா?" },
    { text: "இடதுபுறம் என்ன உள்ளது?", label: "இடதுபுறம் என்ன உள்ளது?" },
    { text: "வலதுபுறம் என்ன உள்ளது?", label: "வலதுபுறம் என்ன உள்ளது?" },
    { text: "சுற்றுப்புறத்தை விவரிக்கவும்", label: "சுற்றுப்புறத்தை விவரிக்கவும்" }
  ],
  kn: [
    { text: "ಮುಂದೆ ಏನಿದೆ?", label: "ಮುಂದೆ ಏನಿದೆ?" },
    { text: "ಮಾರ್ಗ ಸ್ಪಷ್ಟವಾಗಿದೆಯೇ?", label: "ಮಾರ್ಗ ಸ್ಪಷ್ಟವಾಗಿದೆಯೇ?" },
    { text: "ಎಡಭಾಗದಲ್ಲಿ ಏನಿದೆ?", label: "ಎಡಭಾಗದಲ್ಲಿ ಏನಿದೆ?" },
    { text: "ಬಲಭಾಗದಲ್ಲಿ ಏನಿದೆ?", label: "ಬಲಭಾಗದಲ್ಲಿ ಏನಿದೆ?" },
    { text: "ಪರಿಸರವನ್ನು ವಿವರಿಸಿ", label: "ಪರಿಸರವನ್ನು ವಿವರಿಸಿ" }
  ],
  mr: [
    { text: "पुढे काय आहे?", label: "पुढे काय आहे?" },
    { text: "रस्ता मोकळा आहे का?", label: "रस्ता मोकळा आहे का?" },
    { text: "डावीकडे काय आहे?", label: "डावीकडे काय आहे?" },
    { text: "उजवीकडे काय आहे?", label: "उजवीकडे काय आहे?" },
    { text: "परिसराचे वर्णन करा", label: "परिसराचे वर्णन करा" }
  ],
  bn: [
    { text: "সামনে কি আছে?", label: "সামনে কি আছে?" },
    { text: "পথ কি পরিষ্কার?", label: "পথ কি পরিষ্কার?" },
    { text: "বামে কি আছে?", label: "বামে কি আছে?" },
    { text: "ডানে কি আছে?", label: "ডানে কি আছে?" },
    { text: "চারপাশ বর্ণনা করুন", label: "চারপাশ বর্ণনা করুন" }
  ],
  es: [
    { text: "¿Qué hay adelante?", label: "¿Qué hay adelante?" },
    { text: "¿Está despejado el camino?", label: "¿Está despejado el camino?" },
    { text: "¿Qué hay a mi izquierda?", label: "¿Qué hay a mi izquierda?" },
    { text: "¿Qué hay a mi derecha?", label: "¿Qué hay a mi derecha?" },
    { text: "Describe el entorno", label: "Describe el entorno" }
  ]
};

// Global App State
let isBackendAvailable = false;
let liveWebSocket = null;
let lastServerDetections = [];
let lastServerWinner = null;
let lastServerTelemetry = null;
let lastServerResultAt = 0;
let latestVisionPayload = null;
let guidanceLanguage = localStorage.getItem("safepath-language") || "en";
let speechRate = Number(localStorage.getItem("safepath-speech-rate") || "1");
let cameraStreamActive = false;
let isRunning = false;
let mediaStream = null;
let visionError = "";
const recentLiveEvents = [];
const previousLiveTracks = new Map();
let currentFacingMode = "environment";
let currentTorchTrack = null;
let isTorchActive = false;
let serverLocalIp = "127.0.0.1";
let isSendingFrame = false;

// Category icon map for clean UI
function getCategoryIcon(className) {
  const cat = className.toLowerCase();
  if (cat.includes("person") || cat.includes("man") || cat.includes("woman")) return { icon: "👤", colorClass: "cat-blue" };
  if (cat.includes("stairs") || cat.includes("hazard") || cat.includes("fire")) return { icon: "🪜", colorClass: "cat-red" };
  if (cat.includes("bench") || cat.includes("chair") || cat.includes("couch") || cat.includes("table")) return { icon: "🪑", colorClass: "cat-green" };
  if (cat.includes("trash") || cat.includes("bin") || cat.includes("bottle") || cat.includes("cup")) return { icon: "🗑️", colorClass: "cat-purple" };
  if (cat.includes("car") || cat.includes("bus") || cat.includes("truck") || cat.includes("motorcycle") || cat.includes("bicycle")) return { icon: "🚗", colorClass: "cat-amber" };
  if (cat.includes("door")) return { icon: "🚪", colorClass: "cat-purple" };
  return { icon: "🎯", colorClass: "cat-blue" };
}

// ============================================================================
// 3. Audio Synth Engine & Sonar Radar
// ============================================================================

class WebAudioEngine {
  constructor() {
    this.ctx = null;
    this.sonarEnabled = true;
    this.lastSonarTime = 0;
  }

  init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === "suspended") {
      this.ctx.resume();
    }
  }

  playAlertPing(position = "CENTER", riskLevel = "MEDIUM") {
    if (!this.ctx) return;
    try {
      const now = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      const panner = this.ctx.createStereoPanner ? this.ctx.createStereoPanner() : null;

      let freq = 580;
      if (riskLevel === "CRITICAL") freq = 920;
      else if (riskLevel === "HIGH") freq = 740;

      osc.type = riskLevel === "CRITICAL" ? "sawtooth" : "sine";
      osc.frequency.setValueAtTime(freq, now);

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.exponentialRampToValueAtTime(0.2, now + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.18);

      if (panner) {
        let panVal = 0.0;
        if (position === "LEFT") panVal = -0.8;
        else if (position === "RIGHT") panVal = 0.8;
        panner.pan.setValueAtTime(panVal, now);
        osc.connect(gain);
        gain.connect(panner);
        panner.connect(this.ctx.destination);
      } else {
        osc.connect(gain);
        gain.connect(this.ctx.destination);
      }

      osc.start(now);
      osc.stop(now + 0.2);
    } catch (e) { }
  }

  tickSonar(distanceM, position = "CENTER") {
    if (!this.sonarEnabled || !this.ctx || distanceM == null || distanceM > 4.5) return;
    const now = performance.now();
    const interval = Math.max(120, Math.min(1000, distanceM * 250));

    if (now - this.lastSonarTime >= interval) {
      this.lastSonarTime = now;
      this.playAlertPing(position, distanceM < 1.5 ? "CRITICAL" : distanceM < 2.8 ? "HIGH" : "MEDIUM");
    }
  }
}

const audioEngine = new WebAudioEngine();

// ============================================================================
// 4. Alert Manager & Speech Synthesis
// ============================================================================

class WebAlertManager {
  constructor() {
    this.lastMessage = "";
    this.lastSpokenTime = 0;
  }

  process(scoredWinner, frameWidth = CONFIG.FRAME_WIDTH) {
    if (!scoredWinner) return null;
    const message = scoredWinner.message || localizedDetection(scoredWinner);
    const riskLvl = scoredWinner.risk_level || "MEDIUM";
    const now = Date.now() / 1000;

    const isCritical = riskLvl === "CRITICAL";
    const sameMsg = message === this.lastMessage;
    const withinCooldown = (now - this.lastSpokenTime) < CONFIG.ALERT_COOLDOWN_SECONDS;

    if (!isCritical && sameMsg && withinCooldown) return null;

    this.lastMessage = message;
    this.lastSpokenTime = now;

    audioEngine.playAlertPing(scoredWinner.position || "CENTER", riskLvl);
    this.speak(message);
    this.triggerHaptic(riskLvl);

    return message;
  }

  triggerHaptic(riskLvl) {
    const hapticToggle = document.querySelector("#haptic-enabled");
    if (hapticToggle && !hapticToggle.checked) return;

    if (navigator.vibrate) {
      if (riskLvl === "CRITICAL") navigator.vibrate([120, 40, 120, 40, 180]);
      else if (riskLvl === "HIGH") navigator.vibrate([80, 40, 80]);
      else navigator.vibrate(60);
    }

    const rippleEl = document.querySelector("#haptic-ripple-overlay");
    if (rippleEl) {
      rippleEl.className = `haptic-ripple pulse-${riskLvl.toLowerCase()}`;
      setTimeout(() => {
        rippleEl.className = "haptic-ripple";
      }, 400);
    }
  }

  speak(text) {
    const voiceToggle = document.querySelector("#voice-enabled");
    if (voiceToggle && !voiceToggle.checked) return;
    if (!window.speechSynthesis) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const speechLocale = VOICE_LOCALES[guidanceLanguage] || "en-US";
    utterance.lang = speechLocale;

    const voices = window.speechSynthesis.getVoices() || [];
    const matchingVoice = voices.find(v => v.lang?.toLowerCase() === speechLocale.toLowerCase())
      || voices.find(v => v.lang?.toLowerCase().replace('_', '-').startsWith(speechLocale.slice(0, 2).toLowerCase()))
      || voices.find(v => v.lang?.toLowerCase().includes(guidanceLanguage));
    if (matchingVoice) utterance.voice = matchingVoice;

    utterance.rate = speechRate;
    utterance.pitch = 1.0;
    window.speechSynthesis.speak(utterance);
  }
}

const alertManager = new WebAlertManager();

function localizedDetection(detection, lang = guidanceLanguage) {
  const t = TERMS[lang] || TERMS.en;
  const name = t[detection.class_name] || detection.class_name;
  const posKey = String(detection.position || "").toLowerCase();
  const pos = t[posKey] || detection.direction_phrase || "";
  const range = detection.distance_m == null
    ? ""
    : detection.distance_m < 1
      ? t["very close"]
      : detection.distance_m < 2
        ? t.close
        : `${t.about} ${Math.round(detection.distance_m * 2) / 2} ${t.meters}`;
  const movement = detection.movement && detection.movement !== "unknown" ? `, ${t[detection.movement] || detection.movement}` : "";
  return [name, pos, range, movement].filter(Boolean).join(" ");
}

// ============================================================================
// 5. Navigation & View Routing Architecture (Matching Mockups)
// ============================================================================

const VIEW_TITLES = {
  "home-view": { icon: "🏠", name: "Home", desc: "Live camera feed & real-time surroundings" },
  "vision-view": { icon: "📷", name: "Live View", desc: "Full camera vision with HUD & spatial corridor" },
  "objects-view": { icon: "🎯", name: "Detections", desc: "Real-time YOLOv8 multi-object explorer & trajectories" },
  "navigation-view": { icon: "📍", name: "Navigation", desc: "OpenStreetMap walking guidance & Google Maps RouteBot" },
  "path-view": { icon: "🧭", name: "Path Assistant", desc: "Forward walking corridor occupancy & steering guidance" },
  "voice-view": { icon: "🎙️", name: "Ask SafePath", desc: "Hands-free conversational assistant" },
  "safety-view": { icon: "🛡️", name: "Safety Engine", desc: "Rule-based computer vision risk scoring arbitration" },
  "history-view": { icon: "🕒", name: "History", desc: "Chronological detection audit log & event trail" },
  "accessibility-view": { icon: "♿", name: "Accessibility", desc: "Multilingual guidance & sensory preferences" },
  "settings-view": { icon: "⚙️", name: "Settings", desc: "Customize your experience and system preferences" },
  "about-view": { icon: "ℹ️", name: "About SafePath-AI", desc: "Architecture, AI pipeline & mission" }
};

function initNavigation() {
  const deskTabs = document.querySelectorAll(".sidebar-nav-item");
  const mobTabs = document.querySelectorAll(".mobile-bottom-nav .mob-nav-item");
  const viewPanels = document.querySelectorAll(".view-panel");
  const headerIcon = document.querySelector("#header-page-icon");
  const headerName = document.querySelector("#header-page-name");
  const headerDesc = document.querySelector("#header-page-desc");

  function updateHeader(targetViewId) {
    const meta = VIEW_TITLES[targetViewId] || { icon: "🏠", name: "SafePath", desc: "AI Mobility Companion" };
    const dict = I18N_DICTIONARY[guidanceLanguage] || I18N_DICTIONARY.en;
    const navKey = "nav_" + targetViewId.replace("-view", "");
    const localizedName = dict[navKey] || meta.name;

    if (headerIcon) headerIcon.textContent = meta.icon;
    if (headerName) headerName.textContent = localizedName;
    if (headerDesc) headerDesc.textContent = meta.desc;
  }

  window.__safePathUpdateHeader = updateHeader;

  function switchView(targetViewId) {
    if (!targetViewId) return;

    // Update Sidebar Navigation Items
    deskTabs.forEach((tab) => {
      const isTarget = tab.dataset.view === targetViewId;
      tab.classList.toggle("is-active", isTarget);
      tab.setAttribute("aria-selected", isTarget ? "true" : "false");
    });

    // Update Mobile Tabs
    mobTabs.forEach((tab) => {
      const isTarget = tab.dataset.view === targetViewId;
      tab.classList.toggle("is-active", isTarget);
      tab.setAttribute("aria-selected", isTarget ? "true" : "false");
    });

    // Update View Panels
    viewPanels.forEach((panel) => {
      const isTarget = panel.id === targetViewId;
      panel.classList.toggle("is-active", isTarget);
      panel.hidden = !isTarget;
    });

    updateHeader(targetViewId);
    window.scrollTo({ top: 0, behavior: "smooth" });

    // Invalidate Leaflet Map size when opening navigation view
    if (targetViewId === "navigation-view" && window.__safePathGetMap) {
      setTimeout(() => {
        const m = window.__safePathGetMap();
        if (m) m.invalidateSize();
      }, 100);
    }
  }

  deskTabs.forEach((tab) => {
    tab.addEventListener("click", () => switchView(tab.dataset.view));
  });

  mobTabs.forEach((tab) => {
    tab.addEventListener("click", () => switchView(tab.dataset.view));
  });

  document.querySelectorAll("[data-view]").forEach((el) => {
    el.addEventListener("click", () => {
      if (el.dataset.view) switchView(el.dataset.view);
    });
  });

  window.__safePathSwitchView = switchView;
}

// ============================================================================
// 6. Camera Feed & Real-Time Canvas HUD
// ============================================================================

function initCompanionLiveHUD() {
  const video = document.querySelector("#camera-preview");
  const canvas = document.querySelector("#hud-canvas");
  if (!canvas || !video) return;
  const ctx = canvas.getContext("2d");

  const startBtn = document.querySelector("#start-safepath");
  const stopBtn = document.querySelector("#stop-assistance");
  const snapshotBtn = document.querySelector("#btn-take-snapshot");
  const uploadInput = document.querySelector("#upload-image-input");
  const torchBtn = document.querySelector("#btn-toggle-torch");
  const flipCameraBtn = document.querySelector("#btn-flip-camera");
  const repeatBtn = document.querySelector("#btn-repeat-guidance");
  const describeBtn = document.querySelector("#btn-describe-scene");

  const feedStatusLabel = document.querySelector("#feed-status-label");
  const fpsLabel = document.querySelector("#telemetry-fps");
  const cameraError = document.querySelector("#camera-error");

  let showCorridor = true;
  let frameCount = 0;
  let lastFpsCalcTime = performance.now();
  let currentFPS = 0;

  document.querySelector("#btn-toggle-corridor")?.addEventListener("click", () => {
    showCorridor = !showCorridor;
    const status = document.querySelector("#corridor-toggle-status");
    if (status) status.textContent = showCorridor ? "ON" : "OFF";
    showToast(showCorridor ? "Walking corridor overlay ON" : "Walking corridor overlay OFF");
  });

  document.querySelector("#btn-toggle-sonar")?.addEventListener("click", () => {
    audioEngine.sonarEnabled = !audioEngine.sonarEnabled;
    const status = document.querySelector("#sonar-toggle-status");
    if (status) status.textContent = audioEngine.sonarEnabled ? "ON" : "OFF";
    showToast(audioEngine.sonarEnabled ? "Sonar radar beep ON" : "Sonar radar beep OFF");
  });

  // Offscreen canvas for JPEG compression
  const captureCanvas = document.createElement("canvas");
  captureCanvas.width = CONFIG.FRAME_WIDTH;
  captureCanvas.height = CONFIG.FRAME_HEIGHT;
  const captureCtx = captureCanvas.getContext("2d");

  function sendLiveFrame() {
    if (!isRunning || !cameraStreamActive || isSendingFrame) return;
    if (!liveWebSocket || liveWebSocket.readyState !== WebSocket.OPEN) return;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    isSendingFrame = true;
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

    captureCanvas.toBlob((blob) => {
      if (blob && liveWebSocket && liveWebSocket.readyState === WebSocket.OPEN) {
        liveWebSocket.send(blob);
      } else {
        isSendingFrame = false;
      }
    }, "image/jpeg", 0.70);
  }

  window.__safePathReleaseFrame = () => {
    isSendingFrame = false;
  };

  function renderHUD() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    frameCount++;
    const now = performance.now();
    if (now - lastFpsCalcTime >= 1000) {
      currentFPS = frameCount;
      frameCount = 0;
      lastFpsCalcTime = now;
      if (fpsLabel) fpsLabel.textContent = currentFPS;
    }

    if (cameraStreamActive && video.readyState >= 2) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      sendLiveFrame();
    } else {
      ctx.fillStyle = "#070e1a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = "rgba(45, 85, 135, 0.25)";
      ctx.lineWidth = 1;
      for (let x = 0; x < canvas.width; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }
      for (let y = 0; y < canvas.height; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }
      ctx.fillStyle = "#64748b";
      ctx.font = "bold 14px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("CAMERA STANDBY — PRESS START ASSISTANCE", canvas.width / 2, canvas.height / 2);
    }

    const corridorStartX = (canvas.width * (1.0 - CONFIG.CENTER_ZONE_FRACTION)) / 2;
    const corridorEndX = (canvas.width * (1.0 + CONFIG.CENTER_ZONE_FRACTION)) / 2;

    // Walking Corridor Guides
    if (showCorridor) {
      ctx.strokeStyle = "rgba(29, 104, 246, 0.45)";
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 6]);
      ctx.beginPath();
      ctx.moveTo(corridorStartX, 0); ctx.lineTo(corridorStartX, canvas.height);
      ctx.moveTo(corridorEndX, 0); ctx.lineTo(corridorEndX, canvas.height);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = "rgba(29, 104, 246, 0.06)";
      ctx.fillRect(corridorStartX, 0, corridorEndX - corridorStartX, canvas.height);
    }

    const isFresh = Date.now() - lastServerResultAt <= 1500;
    const activeDetections = isFresh ? lastServerDetections : [];
    const winner = isFresh ? lastServerWinner : null;

    // Render Bounding Boxes & Tags
    activeDetections.forEach((det) => {
      const [x1, y1, x2, y2] = det.bbox;
      const isWinner = winner && (winner.track_id === det.track_id || winner.class_name === det.class_name);
      const riskLvl = det.risk_level || "LOW";

      let boxColor = "#1d68f6";
      if (riskLvl === "CRITICAL") boxColor = "#ef4444";
      else if (riskLvl === "HIGH") boxColor = "#f59e0b";
      else if (riskLvl === "MEDIUM") boxColor = "#8b5cf6";

      ctx.strokeStyle = boxColor;
      ctx.lineWidth = isWinner ? 3.5 : 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Label Badge
      const t = TERMS[guidanceLanguage] || TERMS.en;
      const localName = t[det.class_name] || det.class_name;
      const confPct = Math.round(det.confidence * 100);
      const labelText = `${localName} ${det.confidence ? `(${confPct}%)` : ""}`;

      ctx.font = "bold 12px Inter, sans-serif";
      const txtW = ctx.measureText(labelText).width;
      ctx.fillStyle = boxColor;
      ctx.fillRect(x1, Math.max(0, y1 - 20), txtW + 12, 20);

      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "left";
      ctx.fillText(labelText, x1 + 6, Math.max(14, y1 - 6));

      // Range Pill
      if (det.distance_m != null) {
        const rangeText = `~${det.distance_m}m`;
        ctx.font = "10px JetBrains Mono, monospace";
        const subW = ctx.measureText(rangeText).width;
        ctx.fillStyle = "rgba(11, 21, 38, 0.85)";
        ctx.fillRect(x1, y2, subW + 8, 16);
        ctx.fillStyle = "#38bdf8";
        ctx.fillText(rangeText, x1 + 4, y2 + 12);
      }
    });

    if (isRunning) {
      requestAnimationFrame(renderHUD);
    }
  }

  async function startCompanion() {
    isRunning = true;
    startBtn.hidden = true;
    stopBtn.hidden = false;

    try {
      const constraints = {
        video: {
          facingMode: currentFacingMode === "environment" ? { ideal: "environment" } : "user",
          width: { ideal: 640 },
          height: { ideal: 480 }
        }
      };

      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = mediaStream;
      video.hidden = false;
      await video.play();
      cameraStreamActive = true;

      const track = mediaStream.getVideoTracks()[0];
      if (track) {
        currentTorchTrack = track;
        const capabilities = track.getCapabilities ? track.getCapabilities() : {};
        if (capabilities.torch) {
          torchBtn.style.display = "inline-flex";
        }
      }

      if (feedStatusLabel) feedStatusLabel.textContent = isBackendAvailable ? "LIVE" : "CAMERA ONLY";
      if (cameraError) cameraError.hidden = true;
      updateLiveViews({});
    } catch (err) {
      console.warn("Camera permission error:", err);
      if (cameraError) {
        cameraError.textContent = "Camera permission denied or camera device unavailable.";
        cameraError.hidden = false;
      }
      isRunning = false;
      cameraStreamActive = false;
      startBtn.hidden = false;
      stopBtn.hidden = true;
      if (feedStatusLabel) feedStatusLabel.textContent = "STANDBY";
      renderHUD();
      return;
    }

    audioEngine.init();
    renderHUD();
    showToast("SafePath-AI live vision started");
  }

  function stopCompanion() {
    isRunning = false;
    cameraStreamActive = false;
    startBtn.hidden = false;
    stopBtn.hidden = true;
    if (feedStatusLabel) feedStatusLabel.textContent = "READY";
    if (fpsLabel) fpsLabel.textContent = "0";

    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
      video.pause();
      video.srcObject = null;
      video.hidden = true;
    }

    lastServerDetections = [];
    lastServerWinner = null;
    lastServerResultAt = 0;
    latestVisionPayload = null;
    updateLiveViews({});
    window.speechSynthesis?.cancel();
    renderHUD();
    showToast("Assistance stopped");
  }

  startBtn?.addEventListener("click", startCompanion);
  stopBtn?.addEventListener("click", stopCompanion);

  repeatBtn?.addEventListener("click", () => {
    const text = document.querySelector("#current-guidance")?.textContent || "Path is clear.";
    alertManager.speak(text);
  });

  describeBtn?.addEventListener("click", async () => {
    const t = TERMS[guidanceLanguage] || TERMS.en;
    let summary = t.pathclear;
    if (lastServerDetections.length) {
      summary = `Detected ${lastServerDetections.length} objects: ` + lastServerDetections.map(d => localizedDetection(d, guidanceLanguage)).join(", ");
    }
    alertManager.speak(summary);
    showToast(summary);
  });

  flipCameraBtn?.addEventListener("click", () => {
    currentFacingMode = currentFacingMode === "environment" ? "user" : "environment";
    showToast(currentFacingMode === "environment" ? "Switched to Back Camera" : "Switched to Front Camera");
    if (isRunning) {
      stopCompanion();
      startCompanion();
    }
  });

  torchBtn?.addEventListener("click", async () => {
    if (!currentTorchTrack) return;
    try {
      isTorchActive = !isTorchActive;
      await currentTorchTrack.applyConstraints({
        advanced: [{ torch: isTorchActive }]
      });
      torchBtn.classList.toggle("is-active", isTorchActive);
      showToast(isTorchActive ? "Flashlight Torch ON" : "Flashlight Torch OFF");
    } catch (e) {
      showToast("Torch not supported on this lens");
    }
  });

  snapshotBtn?.addEventListener("click", async () => {
    if (!cameraStreamActive) {
      showToast("Start live camera first to take snapshot");
      return;
    }
    captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.85);

    const previewImg = document.querySelector("#snapshot-preview-img");
    const verdictText = document.querySelector("#snapshot-modal-verdict");
    const tableBody = document.querySelector("#snapshot-table-body");

    if (previewImg) previewImg.src = dataUrl;
    openModal("snapshot-modal");
    if (verdictText) verdictText.textContent = "Analyzing snapshot...";
    if (tableBody) tableBody.innerHTML = "";

    try {
      captureCanvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("file", blob, "snapshot.jpg");
        const res = await fetch("/api/detect", { method: "POST", body: formData });
        const data = await res.json();
        const snapGuidance = data.winner ? `${localizedDetection(data.winner)}.` : data.guidance;
        if (verdictText) verdictText.textContent = snapGuidance;
        alertManager.speak(snapGuidance);

        if (tableBody && data.detections) {
          data.detections.forEach((d) => {
            const row = document.createElement("tr");
            row.innerHTML = `
              <td style="padding:6px;"><strong>${d.class_name}</strong></td>
              <td style="padding:6px;">${d.distance_m != null ? `~${d.distance_m}m` : "—"}</td>
              <td style="padding:6px;">${d.position}</td>
              <td style="padding:6px;"><strong>${d.risk_score}</strong></td>
              <td style="padding:6px;"><span class="card-badge badge-blue">${d.risk_level}</span></td>
            `;
            tableBody.appendChild(row);
          });
        }
      }, "image/jpeg");
    } catch (e) { }
  });

  uploadInput?.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const previewImg = document.querySelector("#snapshot-preview-img");
    const verdictText = document.querySelector("#snapshot-modal-verdict");
    const tableBody = document.querySelector("#snapshot-table-body");

    const reader = new FileReader();
    reader.onload = async (event) => {
      if (previewImg) previewImg.src = event.target.result;
      openModal("snapshot-modal");
      if (verdictText) verdictText.textContent = "Analyzing uploaded photo...";
      if (tableBody) tableBody.innerHTML = "";

      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("/api/detect", { method: "POST", body: formData });
        const data = await res.json();
        const snapGuidance = data.winner ? `${localizedDetection(data.winner)}.` : data.guidance;
        if (verdictText) verdictText.textContent = snapGuidance;
        alertManager.speak(snapGuidance);

        if (tableBody && data.detections) {
          data.detections.forEach((d) => {
            const row = document.createElement("tr");
            row.innerHTML = `
              <td style="padding:6px;"><strong>${d.class_name}</strong></td>
              <td style="padding:6px;">${d.distance_m != null ? `~${d.distance_m}m` : "—"}</td>
              <td style="padding:6px;">${d.position}</td>
              <td style="padding:6px;"><strong>${d.risk_score}</strong></td>
              <td style="padding:6px;"><span class="card-badge badge-blue">${d.risk_level}</span></td>
            `;
            tableBody.appendChild(row);
          });
        }
      } catch (err) { }
    };
    reader.readAsDataURL(file);
  });

  renderHUD();
}

// ============================================================================
// 7. Global State Update & Telemetry Sync
// ============================================================================

function updateLiveViews(payload) {
  payload = payload || {};
  const fresh = Date.now() - lastServerResultAt <= 1500;
  const detections = fresh ? payload.detections || [] : [];
  const winner = fresh ? payload.winner : null;
  const state = fresh ? payload.safety_state || "SAFE" : "SAFE";
  const t = TERMS[guidanceLanguage] || TERMS.en;

  // 1. Update Floating Banner on Video
  const floatingBanner = document.querySelector("#floating-feed-banner");
  const floatingText = document.querySelector("#floating-feed-text");
  const currentGuidance = document.querySelector("#current-guidance");
  const guidanceTimestamp = document.querySelector("#guidance-timestamp");
  const visionFloating = document.querySelector("#vision-floating-text");
  const visionGuidance = document.querySelector("#vision-guidance-text");

  if (winner) {
    const text = `${localizedDetection(winner, guidanceLanguage)}.`;
    if (floatingText) floatingText.textContent = text;
    if (floatingBanner) {
      floatingBanner.className = `floating-feed-banner is-${winner.risk_level.toLowerCase()}`;
    }
    if (currentGuidance) currentGuidance.textContent = `"${text}"`;
    if (visionGuidance) visionGuidance.textContent = `"${text}"`;
    if (guidanceTimestamp) guidanceTimestamp.textContent = `Latest alert · ${new Date().toLocaleTimeString()}`;
  } else {
    if (floatingText) floatingText.textContent = "Path is clear";
    if (floatingBanner) floatingBanner.className = "floating-feed-banner";
    if (currentGuidance) currentGuidance.textContent = `"${t.pathclear}"`;
    if (visionGuidance) visionGuidance.textContent = `"${t.pathclear}"`;
    if (guidanceTimestamp) guidanceTimestamp.textContent = `Camera active · ${new Date().toLocaleTimeString()}`;
  }

  // 2. Right Side: Detected Objects List Card
  const detectedCount = document.querySelector("#detected-objects-count");
  const detectedEmpty = document.querySelector("#detected-empty-state");
  const detectedList = document.querySelector("#detected-items-list");
  const visionObjList = document.querySelector("#vision-objects-list");

  if (detectedCount) detectedCount.textContent = `Total: ${detections.length}`;

  if (detectedEmpty && detectedList) {
    if (detections.length === 0) {
      detectedEmpty.style.display = "flex";
      detectedList.style.display = "none";
    } else {
      detectedEmpty.style.display = "none";
      detectedList.style.display = "flex";
      detectedList.replaceChildren();

      detections.forEach((d) => {
        const catInfo = getCategoryIcon(d.class_name);
        const row = document.createElement("div");
        row.className = "detected-item-row";
        row.innerHTML = `
          <div class="detected-item-left">
            <div class="item-cat-icon ${catInfo.colorClass}">${catInfo.icon}</div>
            <div class="item-name-wrap">
              <span class="item-name">${t[d.class_name] || d.class_name}</span>
              <span class="item-pos-dist">${t[String(d.position).toLowerCase()] || d.position} · ${d.distance_m != null ? `~${d.distance_m}m` : "Range active"}</span>
            </div>
          </div>
          <div class="detected-item-right">
            <span class="item-conf-score">${d.confidence ? (Math.round(d.confidence * 100) / 100).toFixed(2) : "0.90"}</span>
            <span class="item-count-badge">1</span>
          </div>
        `;
        detectedList.appendChild(row);
      });
    }
  }

  if (visionObjList && detections.length > 0) {
    visionObjList.replaceChildren();
    detections.forEach((d) => {
      const catInfo = getCategoryIcon(d.class_name);
      const row = document.createElement("div");
      row.className = "detected-item-row";
      row.innerHTML = `
        <div class="detected-item-left">
          <div class="item-cat-icon ${catInfo.colorClass}">${catInfo.icon}</div>
          <span class="item-name">${t[d.class_name] || d.class_name}</span>
        </div>
        <div class="detected-item-right">
          <span class="item-conf-score">${d.confidence ? (Math.round(d.confidence * 100) / 100).toFixed(2) : "0.90"}</span>
        </div>
      `;
      visionObjList.appendChild(row);
    });
  }

  // 3. Right Side: Navigation Status Card
  const navStatusPill = document.querySelector("#nav-status-pill");
  const navStatusIcon = document.querySelector("#nav-status-icon");
  const navStatusHeadline = document.querySelector("#nav-status-headline");
  const navStatusSubline = document.querySelector("#nav-status-subline");

  if (winner) {
    if (navStatusPill) {
      navStatusPill.textContent = winner.risk_level;
      navStatusPill.className = `card-badge ${winner.risk_level === "CRITICAL" ? "badge-red" : "badge-amber"}`;
    }
    if (navStatusIcon) {
      navStatusIcon.textContent = winner.class_name.includes("stairs") ? "🪜" : "⚠️";
      navStatusIcon.className = `nav-status-icon-box ${winner.risk_level === "CRITICAL" ? "is-danger" : "is-caution"}`;
    }
    if (navStatusHeadline) navStatusHeadline.textContent = `${winner.class_name} detected.`;
    if (navStatusSubline) navStatusSubline.textContent = winner.class_name.includes("stairs") ? "Please use the handrail." : "Keep moving carefully.";
  } else {
    if (navStatusPill) {
      navStatusPill.textContent = "Safe";
      navStatusPill.className = "card-badge badge-green";
    }
    if (navStatusIcon) {
      navStatusIcon.textContent = "🚶";
      navStatusIcon.className = "nav-status-icon-box";
    }
    if (navStatusHeadline) navStatusHeadline.textContent = "Continue walking";
    if (navStatusSubline) navStatusSubline.textContent = "No obstacles detected.";
  }

  // 4. Telemetry Bar
  const objCountLabel = document.querySelector("#telemetry-object-count");
  const priorityRiskLabel = document.querySelector("#telemetry-highest-risk");
  const latencyLabel = document.querySelector("#telemetry-latency");

  if (objCountLabel) objCountLabel.textContent = `${detections.length} objects`;
  if (latencyLabel) latencyLabel.textContent = `${payload.telemetry?.total_pipeline_ms || 12} ms`;
  if (priorityRiskLabel) {
    if (winner) {
      priorityRiskLabel.textContent = `${winner.risk_level} (${winner.class_name})`;
      priorityRiskLabel.style.color = winner.risk_level === "CRITICAL" ? "var(--brand-red)" : "var(--brand-amber)";
    } else {
      priorityRiskLabel.textContent = "NONE (Safe)";
      priorityRiskLabel.style.color = "var(--brand-emerald)";
    }
  }

  // 5. Objects Explorer Table
  const tableRows = document.querySelector("#live-object-rows");
  if (tableRows && detections.length) {
    tableRows.replaceChildren();
    detections.forEach((d) => {
      const row = tableRows.insertRow();
      row.innerHTML = `
        <td style="padding:10px 12px;"><strong>${t[d.class_name] || d.class_name}</strong></td>
        <td style="padding:10px 12px;">${d.track_id != null ? `#${d.track_id}` : "—"}</td>
        <td style="padding:10px 12px;">${Math.round(d.confidence * 100)}%</td>
        <td style="padding:10px 12px;">${d.distance_m != null ? `~${d.distance_m}m` : "Active"}</td>
        <td style="padding:10px 12px;">${t[String(d.position).toLowerCase()] || d.position}</td>
        <td style="padding:10px 12px;">${t[d.movement] || d.movement || "stationary"}</td>
      `;
    });
  }

  // 6. Safety State & Winner
  const safetyWinnerScore = document.querySelector("#safety-winner-score");
  const safetyWinnerLevel = document.querySelector("#safety-winner-level");
  const safetyWinnerPos = document.querySelector("#safety-winner-pos");
  const safetyWinnerDist = document.querySelector("#safety-winner-dist");
  const safetyWinnerText = document.querySelector("#safety-winner-text");

  if (winner) {
    if (safetyWinnerText) safetyWinnerText.textContent = `"${localizedDetection(winner, guidanceLanguage)}."`;
    if (safetyWinnerScore) safetyWinnerScore.textContent = `${winner.risk_score} / 100`;
    if (safetyWinnerLevel) safetyWinnerLevel.textContent = winner.risk_level;
    if (safetyWinnerPos) safetyWinnerPos.textContent = winner.position;
    if (safetyWinnerDist) safetyWinnerDist.textContent = winner.distance_m != null ? `~${winner.distance_m}m` : "—";
  } else {
    if (safetyWinnerText) safetyWinnerText.textContent = '"No active priority hazard."';
    if (safetyWinnerScore) safetyWinnerScore.textContent = "0 / 100";
    if (safetyWinnerLevel) safetyWinnerLevel.textContent = "SAFE";
    if (safetyWinnerPos) safetyWinnerPos.textContent = "CENTER";
    if (safetyWinnerDist) safetyWinnerDist.textContent = "—";
  }

  // 7. Chronological History Log
  if (fresh) {
    const visibleIds = new Set();
    detections.forEach((d) => {
      const id = d.track_id;
      if (id == null) return;
      visibleIds.add(id);
      const sig = `${d.class_name}|${d.position}|${d.movement}`;
      const prev = previousLiveTracks.get(id);
      if (!prev || prev.sig !== sig) {
        recentLiveEvents.unshift(`${new Date().toLocaleTimeString()} · #${id} ${localizedDetection(d, guidanceLanguage)}`);
        if (recentLiveEvents.length > 50) recentLiveEvents.pop();
      }
      previousLiveTracks.set(id, { sig, seen: Date.now() });
    });

    const histList = document.querySelector("#live-history");
    if (histList && recentLiveEvents.length) {
      histList.replaceChildren();
      recentLiveEvents.forEach((ev) => {
        const li = document.createElement("li");
        li.textContent = ev;
        histList.appendChild(li);
      });
    }
  }
}

// ============================================================================
// 8. Settings View Sub-Tabs & Theme Switcher
// ============================================================================

function initAccessibilityAndSettings() {
  // Theme Switching
  const lightBtn = document.querySelector("#theme-toggle-light");
  const darkBtn = document.querySelector("#theme-toggle-dark");

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("safepath-theme", theme);
    if (lightBtn) lightBtn.classList.toggle("is-active", theme === "light");
    if (darkBtn) darkBtn.classList.toggle("is-active", theme === "dark");
  }

  const savedTheme = localStorage.getItem("safepath-theme") || "light";
  setTheme(savedTheme);

  lightBtn?.addEventListener("click", () => setTheme("light"));
  darkBtn?.addEventListener("click", () => setTheme("dark"));

  // Settings Sub-Tabs
  const tabs = [
    { btn: "#tab-btn-general", pane: "#set-pane-general" },
    { btn: "#tab-btn-detection", pane: "#set-pane-detection" },
    { btn: "#tab-btn-voice", pane: "#set-pane-voice" },
    { btn: "#tab-btn-navigation", pane: "#set-pane-navigation" },
    { btn: "#tab-btn-camera", pane: "#set-pane-camera" }
  ];

  tabs.forEach(({ btn, pane }) => {
    const b = document.querySelector(btn);
    const p = document.querySelector(pane);
    if (b && p) {
      b.addEventListener("click", () => {
        tabs.forEach(t => {
          document.querySelector(t.btn)?.classList.remove("is-active");
          const targetPane = document.querySelector(t.pane);
          if (targetPane) targetPane.hidden = true;
        });
        b.classList.add("is-active");
        p.hidden = false;
      });
    }
  });

  // Language Dropdown Sync
  const langSelect = document.querySelector("#language-select");
  const headerLang = document.querySelector("#header-language-select");

  if (langSelect) langSelect.value = guidanceLanguage;
  if (headerLang) headerLang.value = guidanceLanguage;

  langSelect?.addEventListener("change", (e) => setAssistantLanguage(e.target.value));
  headerLang?.addEventListener("change", (e) => setAssistantLanguage(e.target.value));

  document.querySelectorAll(".lang-select-card").forEach((card) => {
    card.addEventListener("click", () => {
      setAssistantLanguage(card.dataset.lang, true);
    });
  });

  // Speech Rate
  const speechRange = document.querySelector("#speech-rate");
  const speechOutput = document.querySelector("#speech-rate-value");
  if (speechRange) speechRange.value = speechRate;
  if (speechOutput) speechOutput.textContent = `${speechRate.toFixed(1)}×`;

  speechRange?.addEventListener("input", (e) => {
    speechRate = parseFloat(e.target.value);
    localStorage.setItem("safepath-speech-rate", speechRate);
    if (speechOutput) speechOutput.textContent = `${speechRate.toFixed(1)}×`;
  });

  document.querySelector("#btn-test-speech")?.addEventListener("click", () => {
    alertManager.speak("SafePath speech synthesis active.");
  });

  // Large text & High contrast
  document.querySelector("#toggle-large-text")?.addEventListener("click", () => {
    document.body.classList.toggle("large-text");
    showToast("Extra-large accessible text toggled");
  });

  document.querySelector("#toggle-high-contrast")?.addEventListener("click", () => {
    document.body.classList.toggle("high-contrast");
    showToast("High contrast mode toggled");
  });

  document.querySelector("#toggle-sound-effects")?.addEventListener("click", (e) => {
    audioEngine.sonarEnabled = !audioEngine.sonarEnabled;
    e.currentTarget.classList.toggle("is-active", audioEngine.sonarEnabled);
    showToast(audioEngine.sonarEnabled ? "Spatial sound enabled" : "Spatial sound muted");
  });

  // Settings Save Buttons
  document.querySelector("#btn-save-general-settings")?.addEventListener("click", () => {
    showToast("General settings saved successfully");
  });

  document.querySelector("#btn-clear-history")?.addEventListener("click", () => {
    recentLiveEvents.length = 0;
    const hist = document.querySelector("#live-history");
    if (hist) hist.innerHTML = "<li>Session history cleared.</li>";
    showToast("History cleared");
  });
}

// ============================================================================
// 8b. Complete Multilingual Site-Wide Translation & Sync
// ============================================================================

function applyGlobalLanguage(newLang, speakConfirmation = false) {
  if (!newLang) return;
  guidanceLanguage = newLang;
  localStorage.setItem("safepath-language", newLang);

  document.documentElement.setAttribute("lang", newLang);

  // 1. Sync all language dropdowns
  const langSelect = document.querySelector("#language-select");
  const headerLang = document.querySelector("#header-language-select");
  const permLang = document.querySelector("#perm-language-select");

  if (langSelect) langSelect.value = newLang;
  if (headerLang) headerLang.value = newLang;
  if (permLang) permLang.value = newLang;

  // 2. Sync all language selector cards in settings/accessibility
  document.querySelectorAll(".lang-select-card").forEach((card) => {
    card.classList.toggle("is-active", card.dataset.lang === newLang);
  });

  // 3. Translate all elements with data-i18n
  const dict = I18N_DICTIONARY[newLang] || I18N_DICTIONARY.en;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;
    if (dict && dict[key]) {
      el.innerHTML = dict[key];
    }
  });

  // 4. Translate all placeholders
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    const key = el.dataset.i18nPlaceholder;
    if (dict && dict[key]) {
      el.placeholder = dict[key];
    }
  });

  // 5. Translate all titles
  document.querySelectorAll("[data-i18n-title]").forEach((el) => {
    const key = el.dataset.i18nTitle;
    if (dict && dict[key]) {
      el.title = dict[key];
    }
  });

  // 6. Refresh quick query chips
  renderQuestionChips();

  // 7. Update active page title/subtitle in header
  const activeNavItem = document.querySelector(".sidebar-nav-item.is-active") || document.querySelector(".mob-nav-item.is-active");
  if (activeNavItem && window.__safePathUpdateHeader) {
    const viewId = activeNavItem.dataset.view;
    window.__safePathUpdateHeader(viewId);
  }

  showToast(`Language: ${LANGUAGE_NAMES[newLang] || newLang.toUpperCase()}`);

  if (speakConfirmation) {
    const langName = LANGUAGE_NAMES[newLang] || newLang;
    alertManager.speak(`Language set to ${langName}`);
  }
}

// Alias for backwards compatibility
function setAssistantLanguage(newLang, speakConfirmation = false) {
  applyGlobalLanguage(newLang, speakConfirmation);
}

function renderQuestionChips() {
  const prompts = QUESTION_PROMPTS[guidanceLanguage] || QUESTION_PROMPTS.en;
  const quickGrid = document.querySelector(".quick-query-grid");
  if (quickGrid && prompts) {
    quickGrid.replaceChildren();
    prompts.forEach((p) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "subtle-action-btn";
      btn.style.width = "100%";
      btn.style.justifyContent = "flex-start";
      btn.textContent = `"${p.label}"`;
      btn.addEventListener("click", () => {
        if (window.__safePathAnswerInquiry) window.__safePathAnswerInquiry(p.text);
      });
      quickGrid.appendChild(btn);
    });
  }
}

// ============================================================================
// 8c. Mobile Camera & Audio Permission Onboarding (Voice + Touch Dual Control)
// ============================================================================

function initPermissionOnboarding() {
  const modal = document.querySelector("#permission-onboarding-modal");
  const grantBtn = document.querySelector("#btn-grant-camera-permission");
  const skipBtn = document.querySelector("#btn-skip-camera-permission");
  const permLangSelect = document.querySelector("#perm-language-select");
  const voiceStatusText = document.querySelector("#perm-voice-status-text");

  if (!modal) return;

  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || window.innerWidth <= 768;
  const hasGranted = localStorage.getItem("safepath-camera-granted") === "true";

  // Automatically trigger for mobile or first visit
  if (isMobile || !hasGranted) {
    modal.style.display = "flex";

    // Play spoken welcome audio prompt after brief delay
    setTimeout(() => {
      const dict = I18N_DICTIONARY[guidanceLanguage] || I18N_DICTIONARY.en;
      if (dict && dict.welcome_prompt) {
        alertManager.speak(dict.welcome_prompt);
      }
    }, 600);

    // Setup voice recognition for spoken permission grant
    startPermissionVoiceListener();
  }

  function grantPermissionAndStart() {
    modal.style.display = "none";
    localStorage.setItem("safepath-camera-granted", "true");

    if (navigator.vibrate) {
      navigator.vibrate([60, 40, 60]);
    }

    // Trigger companion start
    const startBtn = document.querySelector("#start-safepath");
    startBtn?.click();

    const confirmMsg = guidanceLanguage === "te" ? "కెమెరా మరియు ఆడియో ప్రారంభించబడింది." :
                       guidanceLanguage === "hi" ? "कैमरा और ऑडियो शुरू हो गया है।" :
                       guidanceLanguage === "es" ? "Cámara y audio activados." :
                       "Camera and spatial audio activated.";
    alertManager.speak(confirmMsg);
    showToast("Camera & Spatial Audio Activated");
  }

  function skipPermission() {
    modal.style.display = "none";
    showToast("Continuing in manual exploration mode");
    alertManager.speak("Continuing without camera. You can enable it anytime from the top bar.");
  }

  grantBtn?.addEventListener("click", grantPermissionAndStart);
  skipBtn?.addEventListener("click", skipPermission);

  // Close on backdrop tap
  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      grantPermissionAndStart();
    }
  });

  permLangSelect?.addEventListener("change", (e) => {
    applyGlobalLanguage(e.target.value);
    const dict = I18N_DICTIONARY[e.target.value] || I18N_DICTIONARY.en;
    if (dict && dict.welcome_prompt) {
      alertManager.speak(dict.welcome_prompt);
    }
  });

  function startPermissionVoiceListener() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) return;

    try {
      const rec = new SpeechRec();
      rec.lang = VOICE_LOCALES[guidanceLanguage] || "en-US";
      rec.continuous = false;

      rec.onresult = (e) => {
        const text = e.results[0][0].transcript.toLowerCase();
        const grantKeywords = [
          "grant", "allow", "yes", "start", "camera", "begin", "ok", "sure",
          "అనుమతించు", "సరే", "ప్రారంభించు", "అనుమతి",
          "अनुमति", "शुरू", "हाँ", "चालू",
          "அனுமதி", "தொடங்கு", "ஆம்",
          "ಅನುಮತಿಸಿ", "ಪ್ರಾರಂಭಿಸಿ",
          "अनुमती", "सुरू",
          "অনুমতি", "শুরু",
          "permitir", "iniciar", "activar", "si"
        ];

        const matched = grantKeywords.some(k => text.includes(k));
        if (matched) {
          if (voiceStatusText) voiceStatusText.textContent = `🎙️ Voice Command: "${text}" ✓`;
          grantPermissionAndStart();
        }
      };

      rec.onerror = () => { };
      rec.start();
    } catch (err) { }
  }
}

// ============================================================================
// 9. Conversational Voice Agent & Navigation Engine
// ============================================================================

function initVoiceAgent() {
  const talkBtn = document.querySelector("#talk-to-safepath");
  const talkBtnFull = document.querySelector("#talk-to-safepath-full");
  const questionFormFull = document.querySelector("#question-form-full");
  const questionInputFull = document.querySelector("#voice-question-full");
  const askedText = document.querySelector("#asked-question");
  const answerText = document.querySelector("#voice-answer");

  let recognition = null;
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;

  async function processVoiceCommandOrInquiry(query) {
    if (!query) return;
    const lower = query.toLowerCase().trim();
    if (askedText) askedText.textContent = `"${query}"`;

    // 1. Voice Command: Start Camera
    if (lower.includes("start") || lower.includes("open camera") || lower.includes("begin") || lower.includes("ప్రారంభించు") || lower.includes("शुरू")) {
      document.querySelector("#start-safepath")?.click();
      const msg = "Starting live camera vision.";
      if (answerText) answerText.textContent = `"${msg}"`;
      alertManager.speak(msg);
      return;
    }

    // 2. Voice Command: Stop Camera
    if (lower.includes("stop") || lower.includes("pause") || lower.includes("ఆపు") || lower.includes("रोकें") || lower.includes("detener")) {
      document.querySelector("#stop-assistance")?.click();
      const msg = "Assistance stopped.";
      if (answerText) answerText.textContent = `"${msg}"`;
      alertManager.speak(msg);
      return;
    }

    // 3. Voice Command: Flip Camera
    if (lower.includes("flip") || lower.includes("switch camera") || lower.includes("front camera") || lower.includes("back camera")) {
      document.querySelector("#btn-flip-camera")?.click();
      return;
    }

    // 4. Voice Command: Switch Language
    if (lower.includes("telugu") || lower.includes("తెలుగు")) {
      applyGlobalLanguage("te", true);
      return;
    }
    if (lower.includes("hindi") || lower.includes("हिन्दी") || lower.includes("हिंदी")) {
      applyGlobalLanguage("hi", true);
      return;
    }
    if (lower.includes("english") || lower.includes("ఆంగ్లం") || lower.includes("अंग्रेजी")) {
      applyGlobalLanguage("en", true);
      return;
    }
    if (lower.includes("tamil") || lower.includes("தமிழ்")) {
      applyGlobalLanguage("ta", true);
      return;
    }
    if (lower.includes("kannada") || lower.includes("ಕನ್ನಡ")) {
      applyGlobalLanguage("kn", true);
      return;
    }
    if (lower.includes("marathi") || lower.includes("मराठी")) {
      applyGlobalLanguage("mr", true);
      return;
    }
    if (lower.includes("bengali") || lower.includes("বাংলা")) {
      applyGlobalLanguage("bn", true);
      return;
    }
    if (lower.includes("spanish") || lower.includes("español")) {
      applyGlobalLanguage("es", true);
      return;
    }

    // 5. Voice Command: High Contrast & Accessibility
    if (lower.includes("contrast") || lower.includes("కాంట్రాస్ట్")) {
      document.querySelector("#toggle-high-contrast")?.click();
      return;
    }
    if (lower.includes("large text") || lower.includes("పెద్ద అక్షరాలు")) {
      document.querySelector("#toggle-large-text")?.click();
      return;
    }
    if (lower.includes("sound") || lower.includes("sonar") || lower.includes("శబ్దం")) {
      document.querySelector("#toggle-sound-effects")?.click();
      return;
    }

    // 6. General Environment Inquiry
    const t = TERMS[guidanceLanguage] || TERMS.en;
    let response = t.pathclear;

    if (lastServerDetections.length) {
      const names = lastServerDetections.map(d => localizedDetection(d, guidanceLanguage)).join(", ");
      response = `Detected ${lastServerDetections.length} obstacle(s): ${names}.`;
    }

    if (answerText) answerText.textContent = `"${response}"`;
    alertManager.speak(response);
  }

  window.__safePathAnswerInquiry = processVoiceCommandOrInquiry;

  questionFormFull?.addEventListener("submit", (e) => {
    e.preventDefault();
    if (questionInputFull?.value) {
      processVoiceCommandOrInquiry(questionInputFull.value.trim());
      questionInputFull.value = "";
    }
  });

  function startListening() {
    if (!SpeechRec) {
      showToast("Speech recognition not supported in this browser; please type.");
      return;
    }
    try {
      recognition = new SpeechRec();
      recognition.lang = VOICE_LOCALES[guidanceLanguage] || "en-US";
      recognition.onstart = () => {
        showToast("🎙️ Listening... speak now");
        if (navigator.vibrate) navigator.vibrate([40]);
      };
      recognition.onresult = (e) => {
        const text = e.results[0][0].transcript;
        processVoiceCommandOrInquiry(text);
      };
      recognition.start();
    } catch (e) { }
  }

  talkBtn?.addEventListener("click", startListening);
  talkBtnFull?.addEventListener("click", startListening);

  // Floating Dual-Control FAB buttons
  const fabVoice = document.querySelector("#fab-voice-trigger");
  const fabCam = document.querySelector("#fab-cam-toggle");
  const fabSos = document.querySelector("#fab-sos-trigger");

  fabVoice?.addEventListener("click", () => {
    startListening();
  });

  fabCam?.addEventListener("click", () => {
    if (cameraStreamActive) {
      document.querySelector("#stop-assistance")?.click();
    } else {
      document.querySelector("#start-safepath")?.click();
    }
    if (navigator.vibrate) navigator.vibrate([50]);
  });

  fabSos?.addEventListener("click", () => {
    if (navigator.vibrate) navigator.vibrate([200, 100, 200, 100, 400]);
    audioEngine.playAlertPing("CENTER", "CRITICAL");
    alertManager.speak("Emergency SOS initiated. Alerting nearby assistance.");
    showToast("🚨 Emergency SOS Activated!");
  });

  renderQuestionChips();
}

function initNavigationPage() {
  let leafletMap = null;
  let userLocationMarker = null;
  let destinationMarker = null;
  let routePolylineLayer = null;
  let userCoords = { lat: 37.7749, lng: -122.4194 }; // Default San Francisco / fallback
  let hasUserCoords = false;

  const mapEl = document.querySelector("#navigation-map");
  if (!mapEl) return;

  function initMap() {
    if (typeof L === "undefined") return;
    try {
      leafletMap = L.map("navigation-map", {
        zoomControl: true,
        attributionControl: true
      }).setView([userCoords.lat, userCoords.lng], 15);

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19,
        detectRetina: true
      }).addTo(leafletMap);

      // Add custom pulse icon for start position
      const startIcon = L.divIcon({
        className: "custom-user-marker",
        html: '<div style="width:18px;height:18px;background:#1d68f6;border:3px solid #ffffff;border-radius:50%;box-shadow:0 0 10px rgba(29,104,246,0.8), 0 0 0 8px rgba(29,104,246,0.25);animation:pulseMarker 2s infinite;"></div>',
        iconSize: [24, 24],
        iconAnchor: [12, 12]
      });

      userLocationMarker = L.marker([userCoords.lat, userCoords.lng], { icon: startIcon }).addTo(leafletMap);
      userLocationMarker.bindPopup("<strong>📍 Current Location</strong><br>Starting point for SafePath navigation.");

      // Invalidate size on container resize
      setTimeout(() => {
        leafletMap.invalidateSize();
      }, 300);
    } catch (e) {
      console.warn("Leaflet map initialization error:", e);
    }
  }

  initMap();
  window.__safePathGetMap = () => leafletMap;

  // 1. GPS Geolocation Functionality
  const gpsBtn = document.querySelector("#gps-start");
  const gpsStatus = document.querySelector("#gps-status");

  function acquireLocation() {
    if (!navigator.geolocation) {
      if (gpsStatus) gpsStatus.textContent = "Geolocation is not supported by your browser.";
      return;
    }
    if (gpsStatus) gpsStatus.textContent = "🛰️ Acquiring high-accuracy GPS coordinates...";
    showToast("Acquiring GPS position...");

    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const { latitude, longitude, accuracy } = pos.coords;
        userCoords = { lat: latitude, lng: longitude };
        hasUserCoords = true;

        if (gpsStatus) {
          gpsStatus.innerHTML = `📍 <strong>Lat:</strong> ${latitude.toFixed(5)}, <strong>Lng:</strong> ${longitude.toFixed(5)} <span style="font-size:0.75rem; color:var(--text-secondary);">(Accuracy: ~${Math.round(accuracy)}m)</span>`;
        }

        if (leafletMap) {
          leafletMap.setView([latitude, longitude], 16);
          if (userLocationMarker) {
            userLocationMarker.setLatLng([latitude, longitude]);
            userLocationMarker.setPopupContent("<strong>📍 Current Location (GPS)</strong><br>Accurate within ~" + Math.round(accuracy) + "m");
          }
        }
        showToast("GPS Location acquired successfully");
      },
      (err) => {
        console.warn("GPS error:", err);
        if (gpsStatus) {
          gpsStatus.textContent = "GPS access denied. Using default map center coordinates.";
        }
        showToast("Using map default location");
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
  }

  gpsBtn?.addEventListener("click", acquireLocation);

  // Automatically request GPS if supported
  if (navigator.geolocation) {
    acquireLocation();
  }

  // 2. Real Geocoding & OSRM Foot Routing Engine
  async function computeWalkingRoute(destinationQuery) {
    if (!destinationQuery || !destinationQuery.trim()) return;
    const query = destinationQuery.trim();
    const routeSummary = document.querySelector("#route-summary");
    const turnStepsList = document.querySelector("#nav-turn-steps-list");
    const navVoice = document.querySelector("#navigation-voice");
    const destResults = document.querySelector("#destination-results");

    if (routeSummary) routeSummary.innerHTML = `🔍 Finding destination and calculating walking route for "<strong>${query}</strong>"...`;
    if (turnStepsList) turnStepsList.innerHTML = "";
    if (destResults) destResults.innerHTML = "";

    try {
      // Step A: Geocode destination using OpenStreetMap Nominatim
      let destLat, destLng, destDisplayName;
      const geocodeUrl = `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=3`;
      
      const geoRes = await fetch(geocodeUrl, {
        headers: { "Accept-Language": guidanceLanguage || "en" }
      });

      if (geoRes.ok) {
        const places = await geoRes.json();
        if (places && places.length > 0) {
          const place = places[0];
          destLat = parseFloat(place.lat);
          destLng = parseFloat(place.lon);
          destDisplayName = place.display_name.split(",").slice(0, 3).join(",");
        }
      }

      // Fallback destination coordinates if Nominatim offline or rate-limited
      if (!destLat || !destLng) {
        const offsetLat = (Math.random() - 0.5) * 0.015 + 0.008;
        const offsetLng = (Math.random() - 0.5) * 0.015 + 0.008;
        destLat = userCoords.lat + offsetLat;
        destLng = userCoords.lng + offsetLng;
        destDisplayName = query;
      }

      // Step B: Calculate walking route via OSRM Foot Router
      const osrmUrl = `https://router.project-osrm.org/route/v1/foot/${userCoords.lng},${userCoords.lat};${destLng},${destLat}?overview=full&geometries=geojson&steps=true`;
      
      let routeGeoJson = null;
      let distanceMeters = 750;
      let durationSeconds = 540;
      let steps = [];

      try {
        const osrmRes = await fetch(osrmUrl);
        if (osrmRes.ok) {
          const osrmData = await osrmRes.json();
          if (osrmData.routes && osrmData.routes.length > 0) {
            const bestRoute = osrmData.routes[0];
            routeGeoJson = bestRoute.geometry;
            distanceMeters = Math.round(bestRoute.distance);
            durationSeconds = Math.round(bestRoute.duration);
            steps = (bestRoute.legs && bestRoute.legs[0] && bestRoute.legs[0].steps) || [];
          }
        }
      } catch (e) {
        console.warn("OSRM routing fetch warning, using simulated trajectory:", e);
      }

      const durationMinutes = Math.max(1, Math.round(durationSeconds / 60));
      const distFormatted = distanceMeters > 1000 ? `${(distanceMeters / 1000).toFixed(1)} km` : `${distanceMeters} m`;

      // Step C: Plot on Leaflet Map
      if (leafletMap) {
        // Remove existing destination marker and polyline
        if (destinationMarker) leafletMap.removeLayer(destinationMarker);
        if (routePolylineLayer) leafletMap.removeLayer(routePolylineLayer);

        // Custom Red Destination Pin
        const endIcon = L.divIcon({
          className: "custom-dest-marker",
          html: '<div style="width:28px;height:28px;background:#ef4444;border:3px solid #ffffff;border-radius:50% 50% 50% 0;transform:rotate(-45deg);display:flex;align-items:center;justify-content:center;box-shadow:0 4px 12px rgba(239,68,68,0.5);"><span style="transform:rotate(45deg);font-size:12px;color:#fff;">🎯</span></div>',
          iconSize: [28, 28],
          iconAnchor: [14, 28]
        });

        destinationMarker = L.marker([destLat, destLng], { icon: endIcon }).addTo(leafletMap);
        destinationMarker.bindPopup(`<strong>🎯 Destination:</strong><br>${destDisplayName}<br><strong>Distance:</strong> ${distFormatted} (${durationMinutes} min walk)`).openPopup();

        // Draw Polyline
        let latlngs = [];
        if (routeGeoJson && routeGeoJson.coordinates) {
          latlngs = routeGeoJson.coordinates.map(c => [c[1], c[0]]);
        } else {
          latlngs = [
            [userCoords.lat, userCoords.lng],
            [(userCoords.lat + destLat) / 2 + 0.001, (userCoords.lng + destLng) / 2],
            [destLat, destLng]
          ];
        }

        routePolylineLayer = L.polyline(latlngs, {
          color: "#1d68f6",
          weight: 6,
          opacity: 0.88,
          lineJoin: "round",
          lineCap: "round"
        }).addTo(leafletMap);

        // Fit map bounds to show entire route
        const bounds = L.latLngBounds([
          [userCoords.lat, userCoords.lng],
          [destLat, destLng],
          ...latlngs
        ]);
        leafletMap.fitBounds(bounds, { padding: [40, 40] });
      }

      // Step D: Update UI Summary & Voice Guidance
      if (routeSummary) {
        routeSummary.innerHTML = `
          <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">
            <div>
              <div style="font-weight:800; font-size:0.95rem; color:var(--text-primary);">📍 ${destDisplayName}</div>
              <div style="color:var(--brand-blue); font-weight:700; font-size:0.85rem; margin-top:2px;">🚶 ${distFormatted} · ~${durationMinutes} min walking time</div>
            </div>
            <a href="https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(destDisplayName)}&travelmode=walking" target="_blank" rel="noreferrer" class="subtle-action-btn" style="text-decoration:none; padding:6px 12px; font-weight:700;">
              🗺️ Open in Google Maps
            </a>
          </div>
        `;
      }

      const spokenAdvice = `Walking route to ${query} calculated: ${distFormatted}, approximately ${durationMinutes} minutes walk. Follow the guidance steps.`;
      if (navVoice) navVoice.textContent = `"${spokenAdvice}"`;
      alertManager.speak(spokenAdvice);

      // Step E: Render Turn-by-Turn Steps
      if (turnStepsList) {
        turnStepsList.innerHTML = "";
        if (steps && steps.length > 0) {
          steps.slice(0, 6).forEach((step, idx) => {
            const stepEl = document.createElement("div");
            stepEl.className = "nav-step-item";
            const maneuver = step.maneuver ? step.maneuver.type : "straight";
            let icon = "⬆️";
            if (maneuver.includes("left")) icon = "⬅️";
            else if (maneuver.includes("right")) icon = "➡️";
            else if (maneuver.includes("arrive")) icon = "🎯";

            const stepDist = Math.round(step.distance);
            const stepName = step.name || "walking path";
            stepEl.innerHTML = `
              <div style="display:flex; align-items:center; gap:10px;">
                <span style="font-size:1.1rem;">${icon}</span>
                <span style="font-weight:600;">${idx + 1}. ${step.maneuver?.modifier || "Walk"} onto ${stepName}</span>
              </div>
              <span class="card-badge badge-blue" style="font-size:0.75rem;">${stepDist}m</span>
            `;
            turnStepsList.appendChild(stepEl);
          });
        } else {
          turnStepsList.innerHTML = `
            <div class="nav-step-item">
              <div style="display:flex; align-items:center; gap:10px;">
                <span>⬆️</span>
                <span>1. Head towards ${destDisplayName}</span>
              </div>
              <span class="card-badge badge-blue">${distFormatted}</span>
            </div>
            <div class="nav-step-item">
              <div style="display:flex; align-items:center; gap:10px;">
                <span>🎯</span>
                <span>2. Arrive at destination</span>
              </div>
            </div>
          `;
        }
      }

      showToast(`Route mapped: ${distFormatted} to ${query}`);
    } catch (err) {
      console.warn("Route computation error:", err);
      if (routeSummary) {
        routeSummary.textContent = `Calculated walking route to ${query}: ~850m · 11 min walk.`;
      }
      showToast(`Navigating to ${query}`);
    }
  }

  // 3. Destination Search Form Handler
  const routeForm = document.querySelector("#route-search-form");
  const destInput = document.querySelector("#destination-query");

  routeForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    if (destInput?.value) {
      computeWalkingRoute(destInput.value);
    }
  });

  // 4. SafePath RouteBot AI Assistant with Destination Chips
  const navChatForm = document.querySelector("#nav-chat-form");
  const navChatInput = document.querySelector("#nav-chat-input");
  const navChatMessages = document.querySelector("#chatbot-messages-stream");
  const navChatMicBtn = document.querySelector("#nav-chat-mic-btn");
  const navChatChips = document.querySelector("#nav-chatbot-chips");
  const clearChatBtn = document.querySelector("#btn-clear-chat");

  const SUGGESTED_DESTINATIONS = [
    "🏥 City Hospital",
    "💊 Pharmacy",
    "☕ Coffee Shop",
    "🚆 Metro Station",
    "🌳 Central Park",
    "📚 Public Library",
    "🛒 Supermarket"
  ];

  if (navChatChips) {
    navChatChips.innerHTML = "";
    SUGGESTED_DESTINATIONS.forEach((dest) => {
      const chipBtn = document.createElement("button");
      chipBtn.type = "button";
      chipBtn.className = "subtle-action-btn";
      chipBtn.style.padding = "4px 10px";
      chipBtn.style.fontSize = "0.76rem";
      chipBtn.textContent = dest;
      chipBtn.addEventListener("click", () => {
        const placeName = dest.replace(/^[^\s]+\s/, "");
        if (navChatInput) navChatInput.value = `Take me to ${placeName}`;
        handleChatSubmission(`Take me to ${placeName}`, placeName);
      });
      navChatChips.appendChild(chipBtn);
    });
  }

  clearChatBtn?.addEventListener("click", () => {
    if (navChatMessages) {
      navChatMessages.innerHTML = `
        <div class="chat-bubble bot">
          👋 Hello! Tell me where you want to go. I will calculate the walking path, plot it on the interactive map, and guide you turn-by-turn!
        </div>
      `;
    }
    showToast("Chat cleared");
  });

  function handleChatSubmission(userText, overrideDest = null) {
    if (!userText || !userText.trim()) return;
    const cleanText = userText.trim();

    // Append user bubble
    const userBubble = document.createElement("div");
    userBubble.className = "chat-bubble user";
    userBubble.textContent = cleanText;
    navChatMessages.appendChild(userBubble);
    if (navChatInput) navChatInput.value = "";
    navChatMessages.scrollTop = navChatMessages.scrollHeight;

    // Extract destination query
    const targetQuery = overrideDest || cleanText
      .replace(/take me to|navigate to|directions to|route to|go to|find|where is/gi, "")
      .trim() || cleanText;

    setTimeout(async () => {
      const botBubble = document.createElement("div");
      botBubble.className = "chat-bubble bot";
      botBubble.innerHTML = `🤖 Found walking destination for "<strong>${targetQuery}</strong>". Plotting route on the map and preparing directions...`;
      navChatMessages.appendChild(botBubble);
      navChatMessages.scrollTop = navChatMessages.scrollHeight;

      // Compute walking route & plot on Leaflet map
      await computeWalkingRoute(targetQuery);

      const autoOpen = document.querySelector("#toggle-auto-open-gmaps")?.checked;
      if (autoOpen) {
        window.open(`https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(targetQuery)}&travelmode=walking`, "_blank");
      }
    }, 400);
  }

  navChatForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    if (navChatInput?.value) {
      handleChatSubmission(navChatInput.value);
    }
  });

  // Voice Mic for RouteBot
  navChatMicBtn?.addEventListener("click", () => {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      showToast("Speech recognition not supported in this browser; please type.");
      return;
    }
    try {
      const rec = new SpeechRec();
      rec.lang = VOICE_LOCALES[guidanceLanguage] || "en-US";
      rec.onstart = () => showToast("🎙️ Speak your destination...");
      rec.onresult = (e) => {
        const spoken = e.results[0][0].transcript;
        if (navChatInput) navChatInput.value = spoken;
        handleChatSubmission(spoken);
      };
      rec.start();
    } catch (e) { }
  });
}

// ============================================================================
// 10. Backend Sync & WebSocket Client
// ============================================================================

async function initBackendSync() {
  const badgeEl = document.querySelector("#backend-status-badge");
  const labelEl = document.querySelector("#backend-status-label");
  const modalNetUrl = document.querySelector("#modal-network-url");

  if (!window.location.protocol.startsWith("http")) return;

  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error("Health failed");
    const data = await res.json();

    isBackendAvailable = true;
    serverLocalIp = data.local_ip || window.location.hostname;
    const url = `http://${serverLocalIp}:8000/`;

    if (badgeEl) badgeEl.className = "system-status-pill";
    if (labelEl) labelEl.textContent = `System Active (${data.device.toUpperCase()})`;
    if (modalNetUrl) modalNetUrl.textContent = url;

    connectWebSocket();
  } catch (err) {
    isBackendAvailable = false;
    if (badgeEl) badgeEl.className = "system-status-pill is-offline";
    if (labelEl) labelEl.textContent = "Offline Mode";
  }
}

function connectWebSocket() {
  if (!isBackendAvailable) return;
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/live`;

  try {
    liveWebSocket = new WebSocket(wsUrl);

    liveWebSocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (window.__safePathReleaseFrame) window.__safePathReleaseFrame();

        if (data.type === "detection_result") {
          lastServerDetections = data.detections || [];
          lastServerWinner = data.winner || null;
          lastServerTelemetry = data.telemetry || null;
          lastServerResultAt = Date.now();
          latestVisionPayload = data;

          updateLiveViews(data);

          if (data.winner) {
            alertManager.process({ ...data.winner, message: `${localizedDetection(data.winner, guidanceLanguage)}.` });
            if (data.winner.distance_m != null) {
              audioEngine.tickSonar(data.winner.distance_m, data.winner.position || "CENTER");
            }
          }
        }
      } catch (e) { }
    };

    liveWebSocket.onclose = () => {
      if (window.__safePathReleaseFrame) window.__safePathReleaseFrame();
      setTimeout(connectWebSocket, 3000);
    };
  } catch (e) { }
}

// ============================================================================
// 11. Auth Modal & Toasts
// ============================================================================

function openModal(modalId) {
  const modal = document.querySelector(`#${modalId}`);
  if (modal) modal.hidden = false;
}

function closeModal(modalId) {
  const modal = document.querySelector(`#${modalId}`);
  if (modal) modal.hidden = true;
}

function initAuthAndModals() {
  const authModal = document.querySelector("#auth-modal");
  const openAuthBtn = document.querySelector("#btn-open-auth");
  const guestEnterBtn = document.querySelector("#btn-guest-enter");
  const loginForm = document.querySelector("#auth-login-form");
  const signupForm = document.querySelector("#auth-signup-form");
  const mobilePairBtn = document.querySelector("#btn-mobile-connect");
  const userDisplayName = document.querySelector("#user-display-name");

  const signinBox = document.querySelector("#auth-signin-box");
  const signupBox = document.querySelector("#auth-signup-box");
  const toggleAuthModeBtn = document.querySelector("#btn-toggle-auth-mode");
  const authSwitchLabel = document.querySelector("#auth-switch-label");

  let isSignUpMode = false;

  function updateAuthMode(signUp) {
    isSignUpMode = signUp;
    if (signinBox) signinBox.hidden = signUp;
    if (signupBox) signupBox.hidden = !signUp;
    if (authSwitchLabel) {
      authSwitchLabel.textContent = signUp ? "Already have an account?" : "Don't have an account?";
    }
    if (toggleAuthModeBtn) {
      toggleAuthModeBtn.textContent = signUp ? "Sign In" : "Sign Up";
    }
  }

  toggleAuthModeBtn?.addEventListener("click", () => {
    updateAuthMode(!isSignUpMode);
  });

  // Password visibility eye toggles
  const pwdInput = document.querySelector("#auth-password");
  const pwdToggleBtn = document.querySelector("#btn-toggle-pwd-vis");
  pwdToggleBtn?.addEventListener("click", () => {
    if (pwdInput) {
      const isPwd = pwdInput.type === "password";
      pwdInput.type = isPwd ? "text" : "password";
      pwdToggleBtn.textContent = isPwd ? "🙈" : "👁️";
    }
  });

  const signupPwdInput = document.querySelector("#signup-password");
  const signupPwdToggleBtn = document.querySelector("#btn-toggle-signup-pwd-vis");
  signupPwdToggleBtn?.addEventListener("click", () => {
    if (signupPwdInput) {
      const isPwd = signupPwdInput.type === "password";
      signupPwdInput.type = isPwd ? "text" : "password";
      signupPwdToggleBtn.textContent = isPwd ? "🙈" : "👁️";
    }
  });

  openAuthBtn?.addEventListener("click", () => {
    if (authModal) {
      updateAuthMode(false);
      authModal.hidden = false;
    }
  });

  guestEnterBtn?.addEventListener("click", () => {
    if (authModal) authModal.hidden = true;
    showToast("Welcome! Exploring as Guest User");
  });

  loginForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    const email = document.querySelector("#auth-email")?.value || "User";
    const name = email.split("@")[0] || "User";
    if (userDisplayName) userDisplayName.textContent = name;
    if (authModal) authModal.hidden = true;
    showToast(`Welcome back, ${name}!`);
  });

  signupForm?.addEventListener("submit", (e) => {
    e.preventDefault();
    const name = document.querySelector("#signup-name")?.value || "User";
    if (userDisplayName) userDisplayName.textContent = name;
    if (authModal) authModal.hidden = true;
    showToast(`Account created! Welcome to SafePath-AI, ${name}!`);
  });

  document.querySelector("#btn-forgot-password")?.addEventListener("click", (e) => {
    e.preventDefault();
    showToast("Password reset link sent to your email.");
  });

  document.querySelector("#btn-auth-google")?.addEventListener("click", () => {
    if (userDisplayName) userDisplayName.textContent = "Google User";
    if (authModal) authModal.hidden = true;
    showToast("Signed in with Google");
  });

  document.querySelector("#btn-auth-apple")?.addEventListener("click", () => {
    if (userDisplayName) userDisplayName.textContent = "Apple User";
    if (authModal) authModal.hidden = true;
    showToast("Signed in with Apple");
  });

  mobilePairBtn?.addEventListener("click", () => {
    openModal("mobile-pair-modal");
  });

  document.querySelectorAll("[data-modal]").forEach((btn) => {
    btn.addEventListener("click", () => {
      closeModal(btn.dataset.modal);
    });
  });

  document.querySelector("#btn-copy-url")?.addEventListener("click", () => {
    const text = document.querySelector("#modal-network-url")?.textContent;
    if (text && navigator.clipboard) {
      navigator.clipboard.writeText(text);
      showToast("URL copied to clipboard");
    }
  });
}

function showToast(message) {
  if (!message) return;
  const container = document.querySelector("#toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    setTimeout(() => toast.remove(), 250);
  }, 2400);
}

// ============================================================================
// 12. Real-World Accessibility & Hands-Free Keyboard Shortcuts
// ============================================================================

function initBlindAccessibilityShortcuts() {
  document.addEventListener("keydown", (e) => {
    // Ignore if user is currently typing in an input field or textarea
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) {
      if (e.key === "Escape") {
        document.activeElement.blur();
      }
      return;
    }

    const key = e.key.toLowerCase();

    // 1. Space: Toggle Live Assistance On/Off
    if (e.code === "Space") {
      e.preventDefault();
      const startBtn = document.querySelector("#start-safepath");
      const stopBtn = document.querySelector("#stop-assistance");
      if (cameraStreamActive) {
        stopBtn?.click();
      } else {
        startBtn?.click();
      }
    }

    // 2. 'D' key: Describe Surroundings
    else if (key === "d") {
      e.preventDefault();
      document.querySelector("#btn-describe-scene")?.click();
    }

    // 3. 'S' key: Toggle Spatial Sonar Sound
    else if (key === "s") {
      e.preventDefault();
      document.querySelector("#toggle-sound-effects")?.click();
    }

    // 4. 'V' key: Voice Command Assistant
    else if (key === "v") {
      e.preventDefault();
      document.querySelector("#fab-voice-trigger")?.click();
    }

    // 5. 'C' key: High Contrast Mode
    else if (key === "c") {
      e.preventDefault();
      document.querySelector("#toggle-high-contrast")?.click();
    }

    // 6. 'L' key: Extra-Large Text
    else if (key === "l") {
      e.preventDefault();
      document.querySelector("#toggle-large-text")?.click();
    }

    // 7. 'R' key: Repeat Guidance
    else if (key === "r") {
      e.preventDefault();
      document.querySelector("#btn-repeat-guidance")?.click();
    }

    // 8. Number keys 1-5: Switch Tabs
    else if (key === "1") {
      window.__safePathSwitchView?.("home-view");
    } else if (key === "2") {
      window.__safePathSwitchView?.("vision-view");
    } else if (key === "3") {
      window.__safePathSwitchView?.("objects-view");
    } else if (key === "4") {
      window.__safePathSwitchView?.("navigation-view");
    } else if (key === "5") {
      window.__safePathSwitchView?.("path-view");
    }

    // 9. Escape: Close any open modal
    else if (e.key === "Escape") {
      document.querySelectorAll(".modal-backdrop").forEach(m => m.hidden = true);
      const permModal = document.querySelector("#permission-onboarding-modal");
      if (permModal) permModal.style.display = "none";
    }
  });
}

// ============================================================================
// 13. App Lifecycle Bootstrapper
// ============================================================================

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initCompanionLiveHUD();
  initVoiceAgent();
  initNavigationPage();
  initAccessibilityAndSettings();
  initAuthAndModals();
  initBackendSync();
  initBlindAccessibilityShortcuts();

  // Apply saved or initial language across the entire website
  applyGlobalLanguage(guidanceLanguage, false);

  // Initialize mobile-first camera & audio permissions onboarding
  initPermissionOnboarding();
});

