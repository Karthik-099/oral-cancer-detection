import json


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "X-Request-ID" in r.headers


def test_models_registry_hides_nothing_retired(client):
    models = client.get("/api/models").json()["models"]
    by_id = {m["id"]: m for m in models}
    assert by_id["oralnet-b4"]["status"] == "production"
    assert by_id["mobilenet-v2-legacy"]["status"] == "retired"


def test_register_rejects_weak_password(client):
    r = client.post("/api/auth/register", json={"email": "a@example.com", "password": "aaaaaaaaaa", "full_name": "A"})
    assert r.status_code == 422


def test_register_login_me(client):
    body = {"email": "Login@Clinic.example.com", "password": "Screening2026", "full_name": "Dr L"}
    assert client.post("/api/auth/register", json=body).status_code == 201
    assert client.post("/api/auth/register", json=body).status_code == 409
    r = client.post("/api/auth/login", json={"email": "login@clinic.example.com", "password": "Screening2026"})
    assert r.status_code == 200
    token = r.json()["token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).json()
    assert me["email"] == "login@clinic.example.com"
    assert "hashed_password" not in me


def test_lockout_after_failed_logins(client):
    client.post("/api/auth/register", json={"email": "lock@clinic.example.com", "password": "Screening2026", "full_name": "L"})
    for _ in range(5):
        assert client.post("/api/auth/login", json={"email": "lock@clinic.example.com", "password": "nope"}).status_code == 401
    r = client.post("/api/auth/login", json={"email": "lock@clinic.example.com", "password": "Screening2026"})
    assert r.status_code == 423


def test_protected_routes_require_auth(client):
    for method, path in [("get", "/api/cases"), ("get", "/api/dashboard"), ("get", "/api/keys"), ("get", "/api/patients")]:
        assert getattr(client, method)(path).status_code == 401
    assert client.get("/api/cases", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_case_lifecycle(client, auth, oral_jpeg):
    r = client.post("/api/cases", headers=auth,
                    files=[("files", ("a.jpg", oral_jpeg, "image/jpeg")), ("files", ("b.jpg", oral_jpeg, "image/jpeg"))],
                    data={"patient_name": "Jane", "patient_reference": "MRN-1", "patient_age": "52",
                          "patient_sex": "female", "site": "Buccal mucosa",
                          "risk_factors": json.dumps(["tobacco", "areca"])})
    assert r.status_code == 201, r.text
    case = r.json()
    assert case["tier"] == "high" and case["image_count"] == 2
    gradcam = case["images"][0]["files"]["gradcam"]
    assert "sig=" in gradcam and "exp=" in gradcam

    assert client.get(gradcam).status_code == 200
    assert client.get(gradcam.split("?")[0]).status_code == 403

    cid = case["id"]
    assert client.get("/api/cases", headers=auth).json()["total"] == 1
    assert client.get("/api/cases?tier=low", headers=auth).json()["total"] == 0
    assert client.get("/api/cases?q=mrn-1", headers=auth).json()["total"] == 1

    lime = client.post(f"/api/cases/{cid}/lime", headers=auth).json()
    assert "lime" in lime["images"][0]["files"]

    fb = client.put(f"/api/cases/{cid}/feedback", headers=auth,
                    json={"verdict": "agree", "clinical_impression": "opmd", "explanation_rating": 4})
    assert fb.status_code == 200 and fb.json()["feedback"]["verdict"] == "agree"

    assert client.patch(f"/api/cases/{cid}", headers=auth, json={"status": "referred"}).json()["status"] == "referred"

    dash = client.get("/api/dashboard", headers=auth).json()
    assert dash["totals"]["cases"] == 1 and dash["agreement_rate"] == 1.0 and dash["totals"]["referred"] == 1

    csv_text = client.get("/api/cases/export.csv", headers=auth).text
    assert cid in csv_text and "opmd" in csv_text

    patients = client.get("/api/patients", headers=auth).json()["patients"]
    timeline = client.get(f"/api/patients/{patients[0]['id']}/timeline", headers=auth).json()
    assert len(timeline["cases"]) == 1

    actions = [e["action"] for e in client.get("/api/audit", headers=auth).json()["events"]]
    assert "case.create" in actions and "case.feedback" in actions

    assert client.delete(f"/api/cases/{cid}", headers=auth).status_code == 200
    assert client.get(f"/api/cases/{cid}", headers=auth).status_code == 404


def test_cases_are_isolated_between_accounts(client, auth, oral_jpeg):
    cid = client.post("/api/cases", headers=auth, files=[("files", ("a.jpg", oral_jpeg, "image/jpeg"))]).json()["id"]
    other = client.post("/api/auth/register", json={"email": "other@clinic.example.com", "password": "Screening2026",
                                                     "full_name": "O"}).json()["token"]
    h = {"Authorization": f"Bearer {other}"}
    assert client.get(f"/api/cases/{cid}", headers=h).status_code == 404
    assert client.delete(f"/api/cases/{cid}", headers=h).status_code == 404
    assert client.get("/api/cases", headers=h).json()["total"] == 0


def test_rejects_non_images_and_spoofed_types(client, auth):
    r = client.post("/api/cases", headers=auth, files=[("files", ("x.jpg", b"<svg onload=alert(1)>", "image/jpeg"))])
    assert r.status_code == 415
    r = client.post("/api/cases", headers=auth, files=[("files", ("a.jpg", b"\xff\xd8\xff" + b"0" * 100, "image/jpeg"))])
    assert r.status_code == 400


def test_rejects_unknown_site_and_bad_fields(client, auth, oral_jpeg):
    files = [("files", ("a.jpg", oral_jpeg, "image/jpeg"))]
    assert client.post("/api/cases", headers=auth, files=files, data={"site": "<script>"}).status_code == 422
    assert client.post("/api/cases", headers=auth, files=files, data={"patient_age": "500"}).status_code == 422


def test_api_keys(client, auth, oral_jpeg):
    created = client.post("/api/keys", headers=auth, json={"name": "PACS bridge"}).json()
    key = created["key"]
    assert key.startswith("oak_")
    assert "key" not in client.get("/api/keys", headers=auth).json()["keys"][0]
    r = client.get("/api/cases", headers={"X-API-Key": key})
    assert r.status_code == 200
    client.delete(f"/api/keys/{created['id']}", headers=auth)
    assert client.get("/api/cases", headers={"X-API-Key": key}).status_code == 401


def test_logout_all_revokes_tokens(client, auth):
    assert client.post("/api/auth/logout-all", headers=auth).status_code == 200
    assert client.get("/api/auth/me", headers=auth).status_code == 401


def test_leads_and_honeypot(client):
    assert client.post("/api/leads", json={"kind": "waitlist-desktop", "email": "x@example.com"}).status_code == 201
    assert client.post("/api/leads", json={"kind": "spam", "email": "x@example.com"}).status_code == 422
    assert client.post("/api/leads", json={"kind": "demo", "email": "bot@example.com", "website": "x"}).status_code == 201


def test_demo_rate_limited(client, oral_jpeg):
    codes = [client.post("/api/demo/analyze", files={"file": ("a.jpg", oral_jpeg, "image/jpeg")}).status_code
             for _ in range(9)]
    assert codes[:8] == [200] * 8 and codes[8] == 429


def test_file_path_traversal_blocked(client):
    assert client.get("/files/..%2F..%2Fetc%2Fpasswd?exp=9999999999&sig=x").status_code in (400, 403, 404)


def test_review_filter_and_filtered_export(client, auth, oral_jpeg):
    cid = client.post("/api/cases", headers=auth, files=[("files", ("a.jpg", oral_jpeg, "image/jpeg"))],
                      data={"risk_factors": '["tobacco"]'}).json()["id"]
    listed = client.get("/api/cases?review=true", headers=auth).json()
    assert listed["total"] == 1 and listed["cases"][0]["patient"]["risk_factors"] == ["tobacco"]
    assert listed["cases"][0]["high_risk_threshold"] == 0.55
    client.put(f"/api/cases/{cid}/feedback", headers=auth, json={"verdict": "agree"})
    assert client.get("/api/cases?review=true", headers=auth).json()["total"] == 0
    assert cid not in client.get("/api/cases/export.csv?tier=low", headers=auth).text
    assert cid in client.get("/api/cases/export.csv?tier=high", headers=auth).text


def test_annotations_saved_validated_and_editable(client, auth, oral_jpeg):
    marks = [{"index": 0, "shapes": [
        {"type": "rect", "label": "Lip", "points": [[0.1, 0.1], [0.4, 0.3]]},
        {"type": "polygon", "label": "Lesion", "points": [[0.5, 0.5], [0.7, 0.5], [0.6, 1.4]]},
    ]}]
    r = client.post("/api/cases", headers=auth, files=[("files", ("a.jpg", oral_jpeg, "image/jpeg"))],
                    data={"annotations": json.dumps(marks)})
    assert r.status_code == 201, r.text
    case = r.json()
    shapes = case["images"][0]["annotations"]
    assert [s["label"] for s in shapes] == ["Lip", "Lesion"]
    assert shapes[1]["points"][2] == [0.6, 1.0]  # clamped into the image
    assert case["site"] == "Lip"  # first labelled site fills the case site

    bad = [{"index": 0, "shapes": [{"type": "rect", "label": "<script>", "points": [[0, 0], [1, 1]]}]}]
    r = client.post("/api/cases", headers=auth, files=[("files", ("a.jpg", oral_jpeg, "image/jpeg"))],
                    data={"annotations": json.dumps(bad)})
    assert r.status_code == 422

    edit = [{"index": 0, "shapes": [{"type": "rect", "label": "Tongue (tip)", "points": [[0.2, 0.2], [0.3, 0.3]]}]}]
    r = client.put(f"/api/cases/{case['id']}/annotations", headers=auth, json={"annotations": edit})
    assert r.status_code == 200 and r.json()["images"][0]["annotations"][0]["label"] == "Tongue (tip)"
