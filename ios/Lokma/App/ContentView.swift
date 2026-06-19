// ContentView — live AR camera with a calorie overlay.
import ARKit
import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = LokmaViewModel()

    var body: some View {
        ZStack(alignment: .bottom) {
            ARViewContainer(session: viewModel.capture.session)
                .ignoresSafeArea()

            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("calib: \(viewModel.calibrationSource)")
                    Spacer()
                    Text(String(format: "conf %.2f", viewModel.calibrationConfidence))
                }
                .font(.caption.monospaced())
                .foregroundStyle(.white.opacity(0.85))

                ForEach(viewModel.labels, id: \.self) { label in
                    Text(label)
                        .font(.headline.monospaced())
                        .foregroundStyle(.white)
                }

                Text(viewModel.status)
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.6))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding()
            .background(.black.opacity(0.45))
        }
        .overlay(alignment: .top) {
            if let guidance = viewModel.guidance {
                GuidanceBanner(guidance: guidance)
                    .padding(.horizontal)
                    .padding(.top, 8)
                    .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.2), value: viewModel.guidance)
        .onAppear { viewModel.start() }
        .onDisappear { viewModel.stop() }
    }
}

/// Live, actionable instruction banner driven by the estimation math gates.
private struct GuidanceBanner: View {
    let guidance: CaptureGuidance

    private var background: Color { guidance.level == .error ? .red : .yellow }
    private var foreground: Color { guidance.level == .error ? .white : .black }

    var body: some View {
        Text(guidance.message)
            .font(.subheadline.weight(.semibold))
            .multilineTextAlignment(.center)
            .foregroundStyle(foreground)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(RoundedRectangle(cornerRadius: 14).fill(background.opacity(0.92)))
            .shadow(radius: 4, y: 2)
    }
}

/// Shows the AR camera feed by attaching the view model's ARSession to an ARSCNView.
struct ARViewContainer: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView(frame: .zero)
        view.session = session
        view.automaticallyUpdatesLighting = true
        return view
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}
