"""Builds customer_analytics.ipynb from cells below, then execute with nbconvert."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Customer Analytics for an Indian D2C Beauty Brand\n"
       "RFM segmentation · cohort retention · K-means · COD/RTO economics · LTV:CAC · predictive CLV\n\n"
       "> Synthetic data (`src/generate_data.py`, seed 21): 60K customers, 120K orders, Jan-2024 to Dec-2025."),
    code("import sys, warnings\nsys.path.insert(0, '../src')\nwarnings.filterwarnings('ignore')\n"
         "import pandas as pd\nfrom IPython.display import Image, display\nfrom analytics import *\nimport charts\n"
         "pd.set_option('display.width', 160); pd.set_option('display.max_columns', 20)\n"
         "customers, orders, cac = load()\norders.head()"),
    md("## 1. Business overview"),
    code("k = kpi_summary(customers, orders)\nprint('Customers with 2+ orders:', k.attrs['repeat_rate_pct'], '%')\nk"),
    md("## 2. Cohort retention\nEach row is the month of a customer's first order; cells show % ordering again N months later."),
    code("cm = cohort_matrix(orders)\ncharts.cohort_heatmap(cm)\ndisplay(Image('../images/cohort_heatmap.png'))"),
    md("### Do sale-acquired customers stick?"),
    code("ret_sale = retention_by_group(orders, customers, 'acquired_in_sale')\ncharts.sale_vs_fullprice(ret_sale)\n"
         "display(ret_sale.rename(index={0: 'full price', 1: 'in a sale'}))\ndisplay(Image('../images/sale_vs_fullprice_retention.png'))"),
    code("retention_by_group(orders, customers, 'acquisition_channel').sort_values('repeat_6m_pct', ascending=False)"),
    md("## 3. RFM segmentation and action playbook\n"
       "Recency and monetary are scored on quintiles. Frequency uses fixed thresholds (1 / 2 / 3 / 4-5 / 6+ orders) because most "
       "customers have exactly one order, which breaks quintiles."),
    code("r = rfm(orders)\nseg = segment_summary(r)\ncharts.segment_chart(seg)\ndisplay(seg)\ndisplay(Image('../images/rfm_segments.png'))"),
    code("# export the CRM target list\nr.reset_index().to_csv('../reports/rfm_customer_segments.csv', index=False)\n"
         "seg.to_csv('../reports/segment_playbook.csv')\nr.segment.value_counts()"),
    md("### Does unsupervised clustering agree?\nK-means on log-scaled, standardised R/F/M; k chosen by silhouette score."),
    code("sil, prof, labels = kmeans_segments(r)\ndisplay(sil.to_frame())\ndisplay(prof)\n"
         "pd.crosstab(r.segment, labels, normalize='index').round(2)"),
    md("## 4. Cash-on-Delivery and RTO\nRTO = parcel refused at the door and shipped back. The brand pays shipping both ways and earns nothing."),
    code("t = rto_analysis(orders, customers)\ncharts.rto_chart(t)\ndisplay(t)\ndisplay(Image('../images/rto_by_tier.png'))"),
    code("# Scenario: Rs 40 off for paying online shifts 30% of COD orders to prepaid\nprepaid_nudge_scenario(orders, shift=0.30, incentive=40)"),
    code("pd.DataFrame([prepaid_nudge_scenario(orders, shift=s, incentive=i) for s in (0.15, 0.3, 0.45) for i in (25, 40, 60)])"
         "[['orders_shifted', 'rto_avoided', 'incentive_cost_lakh', 'net_benefit_lakh']]"
         ".set_index(pd.MultiIndex.from_product([[0.15, 0.3, 0.45], [25, 40, 60]], names=['shift', 'incentive_rs']))"),
    md("## 5. LTV:CAC by acquisition channel\nCAC values are planning assumptions (`data/raw/channel_cac.csv`). LTV = 12-month contribution margin "
       "after product cost, shipping, COD fees, RTO and returns."),
    code("ltv = ltv_by_channel(orders, customers, cac)\ncharts.ltv_cac_chart(ltv)\ndisplay(ltv)\ndisplay(Image('../images/ltv_cac.png'))"),
    md("## 6. Predictive CLV\nFeatures from each customer's **first 90 days**. Target is delivered spend in the **next 180 days**. "
       "This is useful for deciding, early, who deserves retention spend."),
    code("X, y = clv_features(orders, customers)\nmodel, metrics, lift = train_clv_model(X, y)\ncharts.clv_lift_chart(lift, metrics)\n"
         "display(metrics)\ndisplay(Image('../images/clv_model_lift.png'))"),
    md("## 7. Replenishment timing\nWhen should the 'running low?' reminder go out? Days between consecutive orders, by first category."),
    code("inter_purchase_days(orders)"),
    md("## Recommendations\nSee the README for the summary table."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "customer_analytics.ipynb")
