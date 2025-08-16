# PRE-DRAFT → show only Draft Assistant and STOP
if draft_status != "postdraft":
    st.info("📝 **Pre-draft** detected. Roster, Start/Sit, Waivers, Trades, and Scheduler unlock after your draft.")
    st.subheader("Draft Assistant (Pre-draft)")
    st.caption("Upload projections/ADP CSV. Required: `name`, `position`, `proj_points`. Optional: `team`, `adp`, `ecr`.")
    uploaded_predraft = st.file_uploader("Upload projections CSV", type=["csv"], key="draft_upload_predraft")
    if uploaded_predraft:
        try:
            proj = pd.read_csv(uploaded_predraft)
            cols = {c.lower().strip(): c for c in proj.columns}
            need = ["name","position","proj_points"]
            if not all(k in cols for k in need):
                st.error(f"CSV must include: {need}. Found: {list(proj.columns)}")
            else:
                dfp = proj.rename(columns={
                    cols["name"]:"name",
                    cols["position"]:"position",
                    cols["proj_points"]:"proj_points"
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
                    labels = []
                    for i in g.index:
                        labels.append("T1" if i in t1 else "T2" if i in t2 else "T3+")
                    tiers.append(g.assign(tier=labels))
                tiers_df = pd.concat(tiers).sort_values(
                    ["position","tier","proj_points"], ascending=[True, True, False]
                )
                st.dataframe(tiers_df.reset_index(drop=True), use_container_width=True)
                # Keep a copy for AI sandbox
                st.session_state["_predraft_dfp"] = dfp

                st.divider()
                st.subheader("AI Sandbox (works pre-draft)")
                st.caption("💡 This sandbox is available only in pre-draft to test AI reasoning with your uploaded CSV.")
                st.caption("Ask OpenAI about this board. Uses the uploaded CSV + basic heuristics.")
                q = st.text_area("Question", "Who are the best RB values after round 6?")
                if st.button("Ask OpenAI about this CSV"):
                    # Build light summaries for grounding
                    summaries = []
                    try:
                        # top-5 by position
                        for pos_name, grp in dfp.groupby(dfp["position"].str.upper()):
                            top5 = grp.sort_values("proj_points", ascending=False).head(5)
                            names = ", ".join(top5["name"].tolist())
                            summaries.append(f"Top {pos_name}: {names}")
                        # sleepers by ADP if provided
                        if "adp" in dfp.columns:
                            sleepers = (
                                dfp[dfp["proj_points"] >= dfp["proj_points"].quantile(0.75)]
                                .sort_values("adp", ascending=False)  # later ADP = potential value
                                .head(8)["name"].tolist()
                            )
                            if sleepers:
                                summaries.append("Possible sleepers (high proj, late ADP): " + ", ".join(sleepers))
                    except Exception as e:
                        summaries.append(f"CSV summaries unavailable (parse issue: {e})")

                    chosen = select_context(q, summaries, k=3)
                    structured = {"mode": "predraft", "rows": int(len(dfp))}
                    ans = answer_with_ai(q, chosen, structured)
                    st.markdown("### AI Answer")
                    st.write(ans)

                st.caption("Heuristic tiers. Sort/filter to build your queue.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
    else:
        st.info("Tip: export projections from your favorite site and upload here.")
    st.stop()  # hide everything else until postdraft
