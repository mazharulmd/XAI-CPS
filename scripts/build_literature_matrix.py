import csv
from pathlib import Path
import openpyxl
root=Path('.')
wb=openpyxl.load_workbook(root/'XAI_IDS_CPS_Papers_Final.xlsx',data_only=True,read_only=True)
ws=wb['Papers']
headers=list(next(ws.iter_rows(min_row=4,max_row=4,values_only=True)))
rows=[]
for row in ws.iter_rows(min_row=5,max_row=37,values_only=True):
    if row[0] is None: continue
    ident=str(row[0])
    pdf_id='31' if ident=='E1' else ident
    has_pdf=(root/'Research Papers'/f'{pdf_id}.pdf').exists()
    rec={headers[i]:row[i] for i in range(len(headers))}
    rec['PDF file']=f'{pdf_id}.pdf' if has_pdf else ''
    rec['Full-text status']='PDF supplied; title/abstract/conclusion extracted' if has_pdf else 'No matching PDF supplied; workbook summary only'
    rows.append(rec)
out=root/'docs/research/literature_matrix.csv'
out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
    writer.writeheader(); writer.writerows(rows)
print('records',len(rows),'pdfs',sum(bool(r['PDF file']) for r in rows),'without_pdf',[r['#'] for r in rows if not r['PDF file']])
print(out)
