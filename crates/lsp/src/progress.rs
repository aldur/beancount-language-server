use super::LspServerState;

#[derive(Debug, Eq, PartialEq)]
pub(crate) enum Progress {
    Begin,
    Report,
    End,
}

impl Progress {
    /// Builds a fractional progress value, clamped to 0.0..=1.0.
    ///
    /// Clamped rather than asserted: this runs on the main loop, so a
    /// background sender that miscounts would otherwise kill the server over
    /// a cosmetic progress value.
    pub(crate) fn fraction(done: usize, total: usize) -> f64 {
        (done as f64 / total.max(1) as f64).clamp(0.0, 1.0)
    }
}

impl LspServerState {
    // Reports progress to the user via the `WorkDoneProgress` protocol.
    pub(crate) fn report_progress(
        &mut self,
        title: &str,
        state: Progress,
        message: Option<String>,
        fraction: Option<f64>,
        token_suffix: Option<&str>,
    ) {
        // TODO: Ensure that the client supports WorkDoneProgress

        // The clamp matters: `as u32` saturates a negative fraction to 0 and
        // a >1 fraction to a nonsense percentage. (The previous
        // `(0.0..=1.0).contains(&f);` computed a bool and discarded it.)
        let percentage = fraction.map(|f| (f.clamp(0.0, 1.0) * 100.0) as u32);
        let token_label = match token_suffix {
            Some(suffix) => format!("beancount/{title}{suffix}"),
            None => format!("beancount/{title}"),
        };
        let token = lsp_types::ProgressToken::String(token_label);
        let work_done_progress = match state {
            Progress::Begin => {
                self.send_request::<lsp_types::WorkDoneProgressCreateRequest>(
                    lsp_types::WorkDoneProgressCreateParams {
                        token: token.clone(),
                    },
                    |_, _| (),
                );

                serde_json::to_value(lsp_types::WorkDoneProgressBegin {
                    title: title.into(),
                    cancellable: None,
                    message,
                    percentage,
                })
                .expect("Progress should be serializable")
            }
            Progress::Report => serde_json::to_value(lsp_types::WorkDoneProgressReport {
                cancellable: None,
                message,
                percentage,
            })
            .expect("Progress should be serializable"),
            Progress::End => serde_json::to_value(lsp_types::WorkDoneProgressEnd { message })
                .expect("Progress should be serializable"),
        };
        self.send_notification::<lsp_types::ProgressNotification>(lsp_types::ProgressParams {
            token,
            value: work_done_progress,
        });
    }
}
/*pub async fn progress_begin(client: &Client, title: &str) -> ProgressToken {
    let token = NumberOrString::String(format!("beancount-language-server/{}", title));
    let begin = WorkDoneProgressBegin {
        title: title.to_string(),
        cancellable: Some(false),
        message: None,
        percentage: Some(100),
    };

    client
        .send_request::<WorkDoneProgressCreate>(WorkDoneProgressCreateParams {
            token: token.clone(),
        })
        .await
        .unwrap();

    client
        .send_notification::<Progress>(ProgressParams {
            token: token.clone(),
            value: ProgressParamsValue::WorkDone(WorkDoneProgress::Begin(begin)),
        })
        .await;
    token
}

pub async fn progress(client: &Client, token: ProgressToken, message: String) {
    let step = WorkDoneProgressReport {
        cancellable: Some(false),
        message: Some(message),
        percentage: None, //Some(pcnt),
    };
    client
        .send_notification::<Progress>(ProgressParams {
            token,
            value: ProgressParamsValue::WorkDone(WorkDoneProgress::Report(step)),
        })
        .await;
}

pub async fn progress_end(client: &Client, token: ProgressToken) {
    client
        .send_notification::<Progress>(ProgressParams {
            token,
            value: ProgressParamsValue::WorkDone(WorkDoneProgress::End(WorkDoneProgressEnd {
                message: Some("Finished parsing".to_string()),
            })),
        })
        .await;
}
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_fraction_basic() {
        assert_eq!(Progress::fraction(0, 10), 0.0);
        assert_eq!(Progress::fraction(5, 10), 0.5);
        assert_eq!(Progress::fraction(10, 10), 1.0);
    }

    #[test]
    fn test_progress_fraction_edge_cases() {
        // When total is 0, should use max(1) to avoid division by zero
        assert_eq!(Progress::fraction(0, 0), 0.0);

        // Full progress
        assert_eq!(Progress::fraction(100, 100), 1.0);

        // Partial progress
        assert_eq!(Progress::fraction(1, 3), 1.0 / 3.0);
        assert_eq!(Progress::fraction(2, 3), 2.0 / 3.0);
    }

    #[test]
    fn test_progress_fraction_large_numbers() {
        assert_eq!(Progress::fraction(500, 1000), 0.5);
        assert_eq!(Progress::fraction(999, 1000), 0.999);
        assert_eq!(Progress::fraction(1, 1000), 0.001);
    }

    #[test]
    fn test_progress_fraction_clamps_instead_of_panicking() {
        // This runs on the main loop: a miscounting sender must not be able
        // to kill the server over a progress value.
        assert_eq!(Progress::fraction(11, 10), 1.0);
        assert_eq!(Progress::fraction(0, 0), 0.0);
    }

    #[test]
    fn test_progress_enum_variants() {
        let begin = Progress::Begin;
        let report = Progress::Report;
        let end = Progress::End;

        // Just verify they exist and can be created
        assert_eq!(begin, Progress::Begin);
        assert_eq!(report, Progress::Report);
        assert_eq!(end, Progress::End);
    }

    #[test]
    fn test_progress_enum_equality() {
        assert_eq!(Progress::Begin, Progress::Begin);
        assert_eq!(Progress::Report, Progress::Report);
        assert_eq!(Progress::End, Progress::End);

        assert_ne!(Progress::Begin, Progress::Report);
        assert_ne!(Progress::Report, Progress::End);
        assert_ne!(Progress::Begin, Progress::End);
    }
}
