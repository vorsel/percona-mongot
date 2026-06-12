package com.xgen.mongot.embedding.providers.clients;

import com.google.common.annotations.VisibleForTesting;
import com.xgen.mongot.embedding.EmbeddingRequestContext;
import com.xgen.mongot.embedding.MongotMetadata;
import com.xgen.mongot.embedding.VectorOrError;
import com.xgen.mongot.embedding.exceptions.EmbeddingProviderNonTransientException;
import com.xgen.mongot.embedding.exceptions.EmbeddingProviderTransientException;
import com.xgen.mongot.embedding.providers.configs.EmbeddingModelConfig;
import com.xgen.mongot.embedding.providers.configs.EmbeddingServiceConfig;
import com.xgen.mongot.embedding.providers.configs.OpenAiApiSchema;
import com.xgen.mongot.index.definition.quantization.VectorAutoEmbedQuantization;
import com.xgen.mongot.metrics.MetricsFactory;
import com.xgen.mongot.util.bson.JsonCodec;
import com.xgen.mongot.util.bson.parser.BsonDocumentParser;
import com.xgen.mongot.util.bson.parser.BsonParseException;
import com.xgen.mongot.util.concurrent.OneShotSingleThreadExecutor;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.DistributionSummary;
import java.io.IOException;
import java.net.ConnectException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpConnectTimeoutException;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpTimeoutException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import javax.annotation.Nullable;
import javax.net.ssl.SSLException;
import org.bson.BsonDocument;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Per model and service tier client for any server speaking the OpenAI-compatible {@code
 * /v1/embeddings} protocol (OpenAI, Ollama, vLLM, HF TEI, Together, ...).
 *
 * <p>Compared to {@link VoyageClient} this deliberately drops multi-tenant credential routing, flex
 * tier / AIMD congestion control, and billing metadata. It keeps only what every OpenAI-compatible
 * deployment needs: an HTTP/2 client with periodic and on-failure renewal, optional {@code Bearer}
 * auth, base64-float decoding, error categorization, and API-key redaction in logs.
 *
 * <p>Day-1 returns {@code float} vectors only; quantized ({@code scalar}/{@code binary}) requests
 * fail fast with a clear error (client-side quantization is the deferred 5%).
 */
public class OpenAiCompatClient implements ClientInterface {
  private static final Logger LOG = LoggerFactory.getLogger(OpenAiCompatClient.class);
  private static final Duration DEFAULT_TIMEOUT = Duration.ofSeconds(60);

  /** Wall-clock interval after which the {@link HttpClient} is replaced to refresh connections. */
  private static final Duration HTTP_CLIENT_REFRESH_INTERVAL = Duration.ofMinutes(10);

  private static final Duration HTTP_CLIENT_SHUTDOWN_AWAIT = Duration.ofSeconds(5);
  private static final Duration HTTP_CLIENT_SHUTDOWN_NOW_AWAIT = Duration.ofSeconds(2);

  @VisibleForTesting
  public static final String DEFAULT_ENDPOINT = "https://api.openai.com/v1/embeddings";

  private final String modelId;
  private final Optional<MongotMetadata> mongotMetadata;
  private final DistributionSummary inputTokenDistribution;
  private final Counter invalidRequestCounter;

  private URI endpoint;
  private Optional<String> apiKey;

  private volatile HttpClient httpClient;

  /**
   * Epoch millis when {@link #httpClient} was created; compared to {@link
   * #HTTP_CLIENT_REFRESH_INTERVAL} for renewal.
   */
  private volatile long httpClientCreatedEpochMs;

  OpenAiCompatClient(
      EmbeddingModelConfig embeddingModelConfig,
      EmbeddingServiceConfig.ServiceTier tier,
      EmbeddingModelConfig.ConsolidatedWorkloadParams workloadParams,
      MetricsFactory metricsFactory,
      Optional<MongotMetadata> metadata) {
    this.modelId = embeddingModelConfig.name();
    this.mongotMetadata = metadata;
    this.inputTokenDistribution = metricsFactory.summary("inputTokenDistribution");
    this.invalidRequestCounter = metricsFactory.counter("invalidRequestCounter");
    this.httpClient = newHttpClient();
    this.httpClientCreatedEpochMs = System.currentTimeMillis();
    // Assign config fields directly here (rather than via updateConfig) so NullAway can prove the
    // @NonNull fields are initialized on every constructor path.
    this.endpoint = URI.create(workloadParams.providerEndpoint().orElse(DEFAULT_ENDPOINT));
    this.apiKey = extractApiKey(workloadParams.credentials());
    LOG.debug(
        "Initialized OpenAI-compatible client: model={}, endpoint={}, tier={}, apiKey={}",
        this.modelId,
        this.endpoint,
        tier,
        this.apiKey.isPresent() ? "set" : "none");
  }

  @Override
  public List<VectorOrError> embed(List<String> inputs, EmbeddingRequestContext context)
      throws EmbeddingProviderTransientException, EmbeddingProviderNonTransientException {
    // Day-1: float embeddings only. Quantization is performed server-side by the engine and is not
    // expressible in the OpenAI protocol, so fail fast (the deferred 5%).
    if (context.autoEmbedQuantization() != VectorAutoEmbedQuantization.FLOAT) {
      throw new EmbeddingProviderNonTransientException(
          "OPENAI_COMPAT provider currently supports only float embeddings; quantization '"
              + context.autoEmbedQuantization().getName()
              + "' is not supported.");
    }

    // The embeddings endpoint can't handle empty strings, so filter them and back-fill errors.
    List<String> filteredInput = inputs.stream().filter(text -> !text.isEmpty()).toList();
    if (filteredInput.isEmpty()) {
      return inputs.stream().map(ignored -> VectorOrError.EMPTY_INPUT_ERROR).toList();
    }

    LOG.debug(
        "Sending OpenAI-compatible embedding request: model={}, endpoint={}, inputCount={},"
            + " database={}, collection={}",
        this.modelId,
        this.endpoint,
        filteredInput.size(),
        context.database(),
        context.collectionName());

    HttpRequest request;
    try {
      request = buildRequest(filteredInput);
    } catch (IllegalArgumentException e) {
      String message = e.getMessage();
      String cleanedMessage = message != null ? removeApiKeyFromHttpHeader(message) : null;
      IllegalArgumentException cleanedException =
          new IllegalArgumentException(cleanedMessage, e.getCause());
      LOG.error("HTTP Request Error", cleanedException);
      throw new EmbeddingProviderTransientException(cleanedException);
    }

    renewHttpClientIfStale();
    HttpClient clientForRequest = this.httpClient;
    try {
      HttpResponse<String> response =
          clientForRequest.send(request, HttpResponse.BodyHandlers.ofString());
      LOG.debug(
          "Received OpenAI-compatible embedding response: model={}, statusCode={}, inputCount={}",
          this.modelId,
          response.statusCode(),
          filteredInput.size());
      return extractVectorsFromResponse(response, inputs);
    } catch (HttpTimeoutException e) {
      if (e instanceof HttpConnectTimeoutException) {
        renewHttpClientAfterConnectionFailure(e, clientForRequest);
      }
      LOG.error("Got timeout error when sending OpenAI-compatible embedding request", e);
      throw new EmbeddingProviderTransientException(e);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      LOG.error("Got an error when sending OpenAI-compatible embedding request", e);
      throw new EmbeddingProviderTransientException(e);
    } catch (IOException e) {
      if (indicatesConnectionLayerFailure(e)) {
        renewHttpClientAfterConnectionFailure(e, clientForRequest);
      }
      LOG.error("Got an error when sending OpenAI-compatible embedding request", e);
      throw new EmbeddingProviderTransientException(e);
    }
  }

  @Override
  public void updateConfig(EmbeddingModelConfig.ConsolidatedWorkloadParams workloadParams) {
    this.endpoint = URI.create(workloadParams.providerEndpoint().orElse(DEFAULT_ENDPOINT));
    this.apiKey = extractApiKey(workloadParams.credentials());
  }

  private static Optional<String> extractApiKey(
      EmbeddingServiceConfig.EmbeddingCredentials credentials) {
    if (credentials instanceof EmbeddingServiceConfig.OpenAiEmbeddingCredentials openAiCreds) {
      return openAiCreds.apiKey.filter(key -> !key.isBlank());
    }
    return Optional.empty();
  }

  private HttpRequest buildRequest(List<String> inputs) {
    HttpRequest.Builder requestBuilder =
        HttpRequest.newBuilder()
            .uri(this.endpoint)
            .timeout(DEFAULT_TIMEOUT)
            .header("Content-Type", "application/json");

    this.apiKey.ifPresent(key -> requestBuilder.header("Authorization", "Bearer " + key));

    String userAgent =
        this.mongotMetadata
            .map(
                metadata ->
                    String.format(
                        "mongot/%s (%s)", metadata.mongotVersion(), metadata.mongotHostName()))
            .orElse("mongot/UNKNOWN (UNKNOWN)");
    requestBuilder.header("User-Agent", userAgent);

    // The configured `outputDimensions` sizes the index, but we intentionally do NOT forward it as
    // the OpenAI `dimensions` request field: local engines (Ollama/TEI) serve a fixed native
    // dimension and reject the field. Server-side dimension reduction (Matryoshka) is a deferred
    // follow-up.
    BsonDocument body =
        new OpenAiApiSchema.EmbedRequest(
                this.modelId, inputs, OpenAiApiSchema.DEFAULT_ENCODING_FORMAT, Optional.empty())
            .toBson();

    return requestBuilder.POST(HttpRequest.BodyPublishers.ofString(body.toJson())).build();
  }

  private List<VectorOrError> extractVectorsFromResponse(
      HttpResponse<String> response, List<String> inputs)
      throws EmbeddingProviderTransientException, HttpTimeoutException {
    int statusCode = response.statusCode();
    if (statusCode == 400 || statusCode == 422) {
      String errorMessage =
          String.format(
              "Got invalid request, fail fast and give up retries. Response body: %s.",
              response.body());
      LOG.warn(errorMessage);
      this.invalidRequestCounter.increment();
      return inputs.stream().map(ignored -> new VectorOrError(errorMessage)).toList();
    }
    if (statusCode == 429) {
      throw new EmbeddingProviderTransientException(
          String.format("Rate limit exceeded (HTTP 429). Response body: %s", response.body()));
    }
    if (statusCode == 408) {
      throw new HttpTimeoutException(
          String.format("Timeout exception (HTTP 408). Response body: %s", response.body()));
    }
    if (statusCode < 200 || statusCode >= 300) {
      throw new EmbeddingProviderTransientException(
          String.format("Got non OK status from response, status code: %s", statusCode));
    }
    try {
      var embedResponse =
          OpenAiApiSchema.EmbedResponse.fromBson(
              BsonDocumentParser.fromRoot(JsonCodec.fromJson(response.body()))
                  .allowUnknownFields(true)
                  .build());
      embedResponse.usage
          .flatMap(usage -> usage.totalTokens.or(() -> usage.promptTokens))
          .ifPresent(tokens -> this.inputTokenDistribution.record(tokens));
      List<VectorOrError> results = new ArrayList<>();
      var iterator = embedResponse.data.iterator();
      for (String input : inputs) {
        if (input.isEmpty()) {
          results.add(VectorOrError.EMPTY_INPUT_ERROR);
        } else {
          if (!iterator.hasNext()) {
            throw new EmbeddingProviderTransientException(
                "Embedding response returned fewer vectors than non-empty inputs");
          }
          results.add(new VectorOrError(iterator.next().embedding));
        }
      }
      return results;
    } catch (BsonParseException e) {
      throw new EmbeddingProviderTransientException(e);
    }
  }

  /** Redact the API key from error messages. */
  private static String removeApiKeyFromHttpHeader(String message) {
    return message.replaceAll("Bearer [^\"\\s]+", "Bearer <REDACTED-API-KEY>");
  }

  private static HttpClient newHttpClient() {
    return HttpClient.newBuilder().version(HttpClient.Version.HTTP_2).build();
  }

  /**
   * Replaces the {@link HttpClient} periodically so underlying HTTP/2 and TLS connections are not
   * held indefinitely (e.g. DNS updates, server idle limits).
   */
  private void renewHttpClientIfStale() {
    long refreshMs = HTTP_CLIENT_REFRESH_INTERVAL.toMillis();
    if (System.currentTimeMillis() - this.httpClientCreatedEpochMs < refreshMs) {
      return;
    }
    synchronized (this) {
      if (System.currentTimeMillis() - this.httpClientCreatedEpochMs < refreshMs) {
        return;
      }
      replaceHttpClientLocked(false, null);
    }
  }

  /**
   * Whether {@code throwable} or its causes indicate TLS, handshake, or transport connection issues
   * where replacing {@link HttpClient} may help the next retry succeed.
   */
  private static boolean indicatesConnectionLayerFailure(Throwable throwable) {
    for (Throwable t = throwable; t != null; t = t.getCause()) {
      if (t instanceof SSLException || t instanceof ConnectException) {
        return true;
      }
      if (t instanceof IOException) {
        String message = t.getMessage();
        if (message != null) {
          String lower = message.toLowerCase(Locale.ROOT);
          if (lower.contains("connection reset")
              || lower.contains("broken pipe")
              || lower.contains("connection refused")
              || lower.contains("forcibly closed")
              || lower.contains("unexpected end of stream")) {
            return true;
          }
        }
      }
    }
    return false;
  }

  /**
   * Replaces the client after connection/TLS failures so retries do not reuse bad state. If another
   * thread already renewed, {@link #httpClient} will differ and we skip duplicate replace/shutdown.
   */
  private void renewHttpClientAfterConnectionFailure(Throwable cause, HttpClient culpritClient) {
    synchronized (this) {
      if (this.httpClient != culpritClient) {
        return;
      }
      replaceHttpClientLocked(true, cause);
    }
  }

  private void replaceHttpClientLocked(boolean afterConnectionFailure, @Nullable Throwable cause) {
    HttpClient previous = this.httpClient;
    this.httpClient = newHttpClient();
    this.httpClientCreatedEpochMs = System.currentTimeMillis();
    if (afterConnectionFailure) {
      LOG.warn(
          "Renewed OpenAI-compatible HttpClient for model {} after connection/TLS failure: {}",
          this.modelId,
          cause != null ? cause.toString() : "unknown");
    } else {
      LOG.debug(
          "Renewed OpenAI-compatible HttpClient for model {} after {} wall-clock interval",
          this.modelId,
          HTTP_CLIENT_REFRESH_INTERVAL);
    }
    new OneShotSingleThreadExecutor("openai-compat-http-client-shutdown-" + this.modelId)
        .execute(() -> shutdownReplacedHttpClient(previous));
  }

  /**
   * Shuts down the replaced {@link HttpClient} to release its connection pools and threads (JDK 21+
   * {@code HttpClient} lifecycle).
   */
  private void shutdownReplacedHttpClient(HttpClient previous) {
    if (previous == null) {
      return;
    }
    try {
      previous.shutdown();
      if (!previous.awaitTermination(HTTP_CLIENT_SHUTDOWN_AWAIT)) {
        LOG.warn(
            "Previous OpenAI-compatible HttpClient for model {} did not terminate within {};"
                + " forcing shutdownNow",
            this.modelId,
            HTTP_CLIENT_SHUTDOWN_AWAIT);
        previous.shutdownNow();
        if (!previous.awaitTermination(HTTP_CLIENT_SHUTDOWN_NOW_AWAIT)) {
          LOG.warn(
              "Previous OpenAI-compatible HttpClient for model {} still not terminated after"
                  + " shutdownNow",
              this.modelId);
        }
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      LOG.warn(
          "Interrupted while shutting down previous OpenAI-compatible HttpClient for model {}",
          this.modelId,
          e);
      previous.shutdownNow();
    }
  }

  @VisibleForTesting
  void renewHttpClientIfStaleForTesting() {
    renewHttpClientIfStale();
  }

  @VisibleForTesting
  void renewHttpClientAfterConnectionFailureForTesting(Throwable cause, HttpClient culpritClient) {
    renewHttpClientAfterConnectionFailure(cause, culpritClient);
  }

  // For test only.
  @VisibleForTesting
  static void injectHttpClient(OpenAiCompatClient target, HttpClient mockHttpClient) {
    HttpClient previous = target.httpClient;
    target.httpClient = mockHttpClient;
    target.httpClientCreatedEpochMs = System.currentTimeMillis();
    target.shutdownReplacedHttpClient(previous);
  }
}
