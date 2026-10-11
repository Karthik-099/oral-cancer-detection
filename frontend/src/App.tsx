import { lazy, Suspense, useEffect } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import SiteLayout from "@/components/SiteLayout";
import { RequireAuth } from "@/components/AppShell";
import { Toaster } from "@/components/Toast";
import { Spinner } from "@/components/ui";
import { useAuth } from "@/lib/auth";
import { api } from "@/lib/api";
import { syncThresholds } from "@/components/app/kit";
import Home from "@/pages/site/Home";

// Marketing pages other than Home, and the whole app, are code-split.
const Platform = lazy(() => import("@/pages/site/Platform"));
const Models = lazy(() => import("@/pages/site/Models"));
const Deployment = lazy(() => import("@/pages/site/Deployment"));
const Pricing = lazy(() => import("@/pages/site/Pricing"));
const Roadmap = lazy(() => import("@/pages/site/Roadmap"));
const Docs = lazy(() => import("@/pages/site/Docs"));
const Security = lazy(() => import("@/pages/site/Security"));
const About = lazy(() => import("@/pages/site/About"));
const Contact = lazy(() => import("@/pages/site/Contact"));
const Legal = lazy(() => import("@/pages/site/Legal"));
const NotFound = lazy(() => import("@/pages/site/NotFound"));

const Login = lazy(() => import("@/pages/app/Login"));
const Register = lazy(() => import("@/pages/app/Register"));
const Dashboard = lazy(() => import("@/pages/app/Dashboard"));
const NewCase = lazy(() => import("@/pages/app/NewCase"));
const Cases = lazy(() => import("@/pages/app/Cases"));
const CaseDetail = lazy(() => import("@/pages/app/CaseDetail"));
const Patients = lazy(() => import("@/pages/app/Patients"));
const PatientDetail = lazy(() => import("@/pages/app/PatientDetail"));
const Settings = lazy(() => import("@/pages/app/Settings"));
const Report = lazy(() => import("@/pages/app/Report"));

const Fallback = () => <div className="grid min-h-[60vh] place-items-center"><Spinner /></div>;

export default function App() {
  const bootstrap = useAuth((s) => s.bootstrap);
  useEffect(() => { bootstrap(); }, [bootstrap]);
  useEffect(() => {
    api.models().then((m) => syncThresholds(m.threshold, m.high_risk_threshold)).catch(() => { /* defaults apply */ });
  }, []);

  return (
    <BrowserRouter>
      <Suspense fallback={<Fallback />}>
        <Routes>
          <Route element={<SiteLayout />}>
            <Route index element={<Home />} />
            <Route path="platform" element={<Platform />} />
            <Route path="models" element={<Models />} />
            <Route path="deployment" element={<Deployment />} />
            <Route path="pricing" element={<Pricing />} />
            <Route path="roadmap" element={<Roadmap />} />
            <Route path="docs" element={<Docs />} />
            <Route path="security" element={<Security />} />
            <Route path="about" element={<About />} />
            <Route path="contact" element={<Contact />} />
            <Route path="privacy" element={<Legal kind="privacy" />} />
            <Route path="terms" element={<Legal kind="terms" />} />
            <Route path="*" element={<NotFound />} />
          </Route>
          <Route path="login" element={<Login />} />
          <Route path="register" element={<Register />} />
          <Route path="app/cases/:id/report" element={<Report />} />
          <Route path="app" element={<RequireAuth />}>
            <Route index element={<Dashboard />} />
            <Route path="new" element={<NewCase />} />
            <Route path="cases" element={<Cases />} />
            <Route path="cases/:id" element={<CaseDetail />} />
            <Route path="patients" element={<Patients />} />
            <Route path="patients/:id" element={<PatientDetail />} />
            <Route path="settings" element={<Settings />} />
          </Route>
        </Routes>
      </Suspense>
      <Toaster />
    </BrowserRouter>
  );
}
