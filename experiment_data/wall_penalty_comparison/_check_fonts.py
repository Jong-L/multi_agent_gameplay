import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fonts = [f.name for f in fm.fontManager.ttflist]
cjks = [f for f in fonts if any(k in f for k in ['Hei','Song','Ming','Fang','Kai','Yuan','Microsoft','Sim','Noto','WenQuan','Droid','Source'])]
print("CJK fonts found:", sorted(set(cjks)))
