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
        .onAppear { viewModel.start() }
        .onDisappear { viewModel.stop() }
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
