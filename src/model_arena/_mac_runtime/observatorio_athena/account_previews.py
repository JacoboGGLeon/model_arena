"""Miniaturas del explorador 0.9.5, adaptadas a historia y eventos publicados."""
from datetime import date as calendar_date
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw

def spark(dates, values, marked, color, target="saldo"):
    """Escala temporal real, extremos locales y cada observación alertada."""
    pairs=[(str(d)[:10],float(v)) for d,v in zip(dates,values) if v is not None and np.isfinite(float(v))]
    if not pairs:
        canvas=Image.new('RGB',(584,136),'white');out=BytesIO();canvas.save(out,format='PNG');return out.getvalue()
    dates=[d for d,v in pairs];values=np.asarray([v for d,v in pairs])
    marked={str(d)[:10] for d in marked}
    days = [calendar_date.fromisoformat(d).toordinal() for d in dates]
    lo, hi = float(min(values)), float(max(values))
    positions = {0, len(values)-1}
    for part in np.array_split(np.arange(len(values)), min(59, len(values))):
        positions.update([int(part[np.argmin(values[part])]), int(part[np.argmax(values[part])])])
    hits = [i for i, d in enumerate(dates) if d in marked]
    positions.update(hits)
    def xy(i):
        x = 9 + 566*(days[i]-days[0])/max(1,days[-1]-days[0])
        y = 68 if hi==lo else 126-116*(float(values[i])-lo)/(hi-lo)
        return (x,y)
    canvas=Image.new('RGB',(584,136),'white')
    pen=ImageDraw.Draw(canvas)
    points=[xy(i) for i in sorted(positions)]
    if len(points)>1: pen.line(points,fill=color,width=3)
    else: pen.point(points[0],fill=color)
    for index in hits:
        x,y=xy(index)
        pen.ellipse((x-5,y-5,x+5,y+5),fill='#c52c48' if target=='saldo' else 'white',outline='#c52c48' if target=='saldo' else '#15202b',width=2)
    image=BytesIO()
    canvas.save(image,format='PNG')
    return image.getvalue()
