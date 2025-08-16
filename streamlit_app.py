# ───────────────────── Draft Assistant (postdraft tab) ─────────────────────
with tabs[0]:
    st.subheader("Draft Assistant")
    uploaded = st.file_uploader("Upload projections CSV", type=["csv"], key="draft_upload_post")
    if uploaded:
        try:
            proj = pd.read_csv(uploaded)
            cols = {c.lower().strip(): c for c in proj.columns}
            need = ["name","position","proj_points"]
            if not all(k in cols for k in need):
                st.error(f"CSV must include: {need}. Found: {list(proj.columns)}")
            else:
                dfp = proj.rename(columns={
                    cols["name"]: "name",
                    cols["position"]: "position",
                    cols["proj_points"]: "proj_points"
                })
                for opt in ("adp","ecr","team"):
                    if opt in cols:
                        dfp[opt] = proj[cols[opt]]

                tiers = []
                for pos, grp in dfp.groupby(dfp["position"].str.upper()):
                    g = grp.copy()
                    top = 12 if pos in ("WR","RB") else 6 if pos in ("QB","TE") else max(4, len(g)//5)
                    top_sorted = g.sort_values("proj_points", ascending=False)
                    t1 = set(top_sorted.head(top).index)
                    t2 = set(top_sorted.iloc[top:top*2].index)
                    labels = [("T1" if i in t1 else "T2" if i in t2 else "T3+") for i in g.index]
                    tiers.append(g.assign(tier=labels))

                tiers_df = pd.concat(tiers).sort_values(
                    ["position","tier","proj_points"], ascending=[True, True, False]
                )
                st.dataframe(tiers_df.reset_index(drop=True), use_container_width=True)

        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    else:
        st.caption("Upload projections/ADP to populate tiers.")



