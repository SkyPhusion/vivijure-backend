// Jenkins pipeline for the vivijure-backend GHCR worker image.
//
// Mirrors the vivijure-serverless release mechanics: the image + version come from a git TAG on
// HEAD (explicit and immutable), not a parsed commit subject. One release track:
//
//   backend-vX.Y.Z  ->  ghcr.io/skyphusion/vivijure-backend:X.Y.Z (+ :latest)
//
// A commit push with no release tag is a NO-OP. This pipeline BUILDS + PUSHES only; deploying
// (pinning the RunPod endpoint template to a built image) is a separate, deliberate step --
// scripts/pin-runpod-template.py -- so an iteration tag never re-pins the live endpoint by surprise.
//
// Release flow (tag the commit you want released, normally current main HEAD):
//   git push origin main
//   git tag backend-v0.1.0 && git push origin backend-v0.1.0
//
// Jenkins job: a Pipeline-from-SCM (branch main, Script Path Jenkinsfile) with a githubPush()
// trigger; the build context is the REPO ROOT and the Dockerfile is deploy/Dockerfile.
// Credentials: ghcr-skyphusion (username/PAT with write:packages).

pipeline {
    agent { label 'build' }

    options {
        // CUDA + torch cu128 + diffusers is a heavy from-scratch build; a cache-warm rebuild
        // finishes in well under 10 minutes.
        timeout(time: 120, unit: 'MINUTES')
        timestamps()
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '30', artifactNumToKeepStr: '10'))
    }

    environment {
        REGISTRY        = 'ghcr.io'
        OWNER           = 'skyphusion'
        WORKER_IMAGE    = 'vivijure-backend'
        DOCKER_BUILDKIT = '1'
    }

    stages {
        stage('checkout') {
            steps {
                checkout scm
            }
        }

        stage('derive image from tag') {
            steps {
                script {
                    // The branch checkout does NOT fetch tags; fetch before resolving the tag on HEAD.
                    sh 'git fetch --force --tags origin 2>/dev/null || true'
                    def tag = (env.TAG_NAME ?: '').trim()
                    if (!tag) {
                        tag = sh(
                            script: 'git describe --tags --exact-match HEAD 2>/dev/null || true',
                            returnStdout: true,
                        ).trim()
                    }
                    env.REF_TAG = tag
                    env.WORKER_VERSION = ''
                    def m = (tag =~ /^backend-v(\d+\.\d+\.\d+)$/)
                    if (m) {
                        env.WORKER_VERSION = m[0][1]
                    }
                    echo "ref:    ${tag ?: (env.BRANCH_NAME ?: '(unknown)')}"
                    echo "worker: ${env.WORKER_VERSION ?: '-'}"
                }
            }
        }

        stage('login GHCR') {
            when { expression { return env.WORKER_VERSION?.trim() } }
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'ghcr-skyphusion',
                    usernameVariable: 'GHCR_USER',
                    passwordVariable: 'GHCR_TOKEN',
                )]) {
                    sh 'echo "$GHCR_TOKEN" | docker login "$REGISTRY" -u "$GHCR_USER" --password-stdin'
                }
            }
        }

        stage('build + push worker') {
            when { expression { return env.WORKER_VERSION?.trim() } }
            steps {
                // Build context is the repo root so COPY src/... resolves; Dockerfile lives in deploy/.
                sh '''
                    set -eu
                    IMG="${REGISTRY}/${OWNER}/${WORKER_IMAGE}"
                    docker build \
                        --pull \
                        --platform=linux/amd64 \
                        -f deploy/Dockerfile \
                        -t "${IMG}:${WORKER_VERSION}" \
                        -t "${IMG}:latest" \
                        .
                    docker push "${IMG}:${WORKER_VERSION}"
                    docker push "${IMG}:latest"
                '''
            }
        }

        stage('no-op note') {
            when { expression { return !(env.WORKER_VERSION?.trim()) } }
            steps {
                echo "Ref '${env.REF_TAG ?: env.BRANCH_NAME}' is not a backend-v* tag; nothing to build."
            }
        }
    }

    post {
        always {
            sh 'docker logout "$REGISTRY" || true'
            sh 'docker image prune -f --filter "until=168h" || true'
        }
        success {
            script {
                if (env.WORKER_VERSION?.trim()) {
                    echo "Pushed ${env.WORKER_IMAGE}:${env.WORKER_VERSION} to GHCR."
                    echo "Deploy is separate: scripts/pin-runpod-template.py pins the RunPod template to a built image."
                } else {
                    echo "No image pushed (ref was not a release tag)."
                }
            }
        }
        failure {
            echo "Build failed. Check the docker login + build logs above."
            mail to: 'conrad@rockenhaus.net',
                 subject: "FAILED: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                 body: "Build failed: ${env.BUILD_URL}"
        }
    }
}
