# 🚀 Deploy PhytoVision AI to Streamlit Community Cloud

## One-Click Deploy (Easiest - 3 minutes)

### ✅ Prerequisites
- GitHub account (already have it!)
- Streamlit account (free signup at streamlit.io)

### 🎯 Steps

#### 1. **Visit Streamlit Community Cloud**
```
Go to: https://streamlit.io/cloud
Click "Get Started" or "Sign Up"
```

#### 2. **Connect Your GitHub Account**
- Click "New app"
- Select "GitHub" as source
- Connect your GitHub account (if not already)
- Choose this repo: `PhytoVision-AI`

#### 3. **Configure the Deployment**
Fill in these fields:
```
Repository: premsahu-3125/PhytoVision-AI
Branch: main
Main file path: streamlit_app.py
```

#### 4. **Advanced Settings (Important!)**
```
Settings → Advanced settings
├─ Requirements file: requirements-streamlit.txt
├─ Python version: 3.11
└─ Save & Deploy
```

**Why `requirements-streamlit.txt`?**
- CPU-only PyTorch (faster, fits free tier)
- Skips FastAPI, ONNX, matplotlib
- ~80% smaller, deploys in 2-3 minutes

#### 5. **Wait for Deployment**
- Streamlit builds and deploys automatically
- Watch the logs on right side
- ✅ When it says "Your app is ready", click the link!

#### 6. **You Get a Live URL**
```
https://phytovision-ai-[random].streamlit.app
```

#### 7. **Test Your App**
1. Upload a bean leaf image
2. See the disease diagnosis
3. View Grad-CAM heatmap
4. Download PDF report

---

## 🎨 What Your Live App Shows

```
PhytoVision AI Interface
├─ 📸 Upload / Camera / Gallery section
├─ 🤖 Diagnosis results with confidence
├─ 🔥 Grad-CAM heatmap visualization
├─ 💊 Treatment advisory
└─ 📄 PDF download
```

---

## 📌 Add to GitHub About Section

Once deployed, add the live link to your GitHub repo:

### Option 1: GitHub CLI
```bash
gh repo edit premsahu-3125/PhytoVision-AI \
  --homepage "https://phytovision-ai-[random].streamlit.app"
```

### Option 2: Manual (GitHub Web)
1. Go to https://github.com/premsahu-3125/PhytoVision-AI
2. Click ⚙️ **Settings**
3. Scroll to "About" section
4. Paste your Streamlit URL in **Website** field
5. Click **Save changes**

---

## ✨ Final Result

Your GitHub repository will show:
```
PhytoVision-AI
AI-powered bean leaf disease detector...

🔗 Website: https://phytovision-ai-[random].streamlit.app

[Code] [Issues] [Pull Requests]
```

**Click the link → Live app opens → See your project in action!** 🌿

---

## 🆘 Troubleshooting

### "Module not found" errors
- Check `requirements-streamlit.txt` is set in Advanced settings
- Streamlit rebuilds automatically after 30 sec

### "Slow first load"
- First load takes 10-15 seconds (model download)
- Subsequent loads are instant ⚡

### "Image upload not working"
- Check browser permissions for camera
- Try file upload instead

### "Out of memory"
- Streamlit free tier has limits
- Model (PhytoVision) is optimized for CPU
- CPU inference takes ~3-5 seconds per image

---

## 💡 Pro Tips

### Auto-Deploy on Every Push
- Just push to GitHub: `git push`
- Streamlit auto-rebuilds from main branch
- Your live app updates automatically! 🔄

### Share Different URLs
- Create multiple branches → Multiple live URLs
- Great for A/B testing or different versions

### Monitor Performance
- Streamlit dashboard shows usage
- Free tier includes plenty of compute time

---

## 📊 What's Included in Deployment

✅ **Streamlit app** - Full interactive UI  
✅ **Model weights** - Pre-trained MobileNetV4  
✅ **Requirements** - All dependencies  
✅ **Example images** - For testing  
✅ **PDF generation** - Reports work  
✅ **Grad-CAM visualization** - XAI features  

---

## 🎯 Timeline

⏱️ **5-10 minutes** → Full deployment  
⏱️ **1 minute** → Add to GitHub About  
✅ **TOTAL** → Less than 15 minutes!

---

## 🔗 Useful Links

- **Streamlit Docs:** https://docs.streamlit.io
- **Streamlit Community Cloud:** https://streamlit.io/cloud
- **Your App:** https://share.streamlit.io (dashboard)
- **This Repo:** https://github.com/premsahu-3125/PhytoVision-AI

---

**You're all set!** 🚀 Deploy now and share your AI project! 🌿
