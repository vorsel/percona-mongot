package com.xgen.mongot.embedding.providers.clients;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;

import com.xgen.mongot.config.util.DeploymentEnvironment;
import com.xgen.mongot.embedding.EmbeddingRequestContext;
import com.xgen.mongot.embedding.VectorOrError;
import com.xgen.mongot.embedding.exceptions.EmbeddingProviderNonTransientException;
import com.xgen.mongot.embedding.exceptions.EmbeddingProviderTransientException;
import com.xgen.mongot.embedding.providers.configs.EmbeddingModelConfig;
import com.xgen.mongot.embedding.providers.configs.EmbeddingServiceConfig;
import com.xgen.mongot.embedding.providers.configs.EmbeddingServiceConfig.EmbeddingProvider;
import com.xgen.mongot.embedding.providers.configs.EmbeddingServiceConfig.ServiceTier;
import com.xgen.mongot.index.definition.quantization.VectorAutoEmbedQuantization;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Base64;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Flow;
import java.util.concurrent.TimeUnit;
import org.junit.Test;
import org.mockito.ArgumentCaptor;

public class OpenAiCompatClientTest {

  private static final EmbeddingServiceConfig.ErrorHandlingConfig RETRY_CONFIG =
      new EmbeddingServiceConfig.ErrorHandlingConfig(3, 10L, 10L, 0.1);

  private static EmbeddingRequestContext floatContext() {
    return new EmbeddingRequestContext(
        "testdb", "testIndex", "testCollection", 3, VectorAutoEmbedQuantization.FLOAT);
  }

  private static EmbeddingModelConfig openAiModel(Optional<String> apiKey) {
    return openAiModel(apiKey, Optional.empty());
  }

  private static EmbeddingModelConfig openAiModel(
      Optional<String> apiKey, Optional<String> authHeaderName) {
    return openAiModel(
        apiKey,
        authHeaderName,
        new EmbeddingServiceConfig.OpenAiModelConfig(
            Optional.empty(), Optional.of(96), Optional.of(120_000), Optional.empty()));
  }

  private static EmbeddingModelConfig openAiModel(
      Optional<String> apiKey,
      Optional<String> authHeaderName,
      EmbeddingServiceConfig.OpenAiModelConfig modelConfig) {
    EmbeddingServiceConfig.OpenAiEmbeddingCredentials creds =
        new EmbeddingServiceConfig.OpenAiEmbeddingCredentials(apiKey, authHeaderName);
    EmbeddingServiceConfig.EmbeddingConfig config =
        new EmbeddingServiceConfig.EmbeddingConfig(
            Optional.empty(),
            modelConfig,
            RETRY_CONFIG,
            creds,
            Optional.empty(),
            Optional.empty(),
            Optional.empty(),
            Optional.empty(),
            true,
            Optional.of("http://localhost:11434/v1/embeddings"),
            false,
            Optional.empty());
    return EmbeddingModelConfig.create("bge-m3", EmbeddingProvider.OPENAI_COMPAT, config);
  }

  private static OpenAiCompatClient newClient(EmbeddingModelConfig model) {
    EmbeddingClientFactory factory =
        new EmbeddingClientFactory(new SimpleMeterRegistry(), DeploymentEnvironment.COMMUNITY);
    ClientInterface client =
        factory.createEmbeddingClient(model, ServiceTier.QUERY, model.query());
    assertTrue(
        "Expected OPENAI_COMPAT provider to build an OpenAiCompatClient",
        client instanceof OpenAiCompatClient);
    return (OpenAiCompatClient) client;
  }

  private static String base64Floats(float... values) {
    ByteBuffer buffer =
        ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    for (float value : values) {
      buffer.putFloat(value);
    }
    return Base64.getEncoder().encodeToString(buffer.array());
  }

  private static HttpClient mockHttpClient(int statusCode, String body) throws Exception {
    HttpClient mockClient = mock(HttpClient.class);
    HttpResponse<String> mockResponse = mock(HttpResponse.class);
    doReturn(statusCode).when(mockResponse).statusCode();
    doReturn(body).when(mockResponse).body();
    doReturn(mockResponse)
        .when(mockClient)
        .send(any(HttpRequest.class), any(HttpResponse.BodyHandler.class));
    doReturn(true).when(mockClient).awaitTermination(any(Duration.class));
    return mockClient;
  }

  @Test
  public void embed_decodesBase64FloatEmbeddings() throws Exception {
    OpenAiCompatClient client = newClient(openAiModel(Optional.empty()));
    String body =
        String.format(
            "{\"object\":\"list\",\"data\":[{\"embedding\":\"%s\",\"index\":0}],"
                + "\"usage\":{\"total_tokens\":1}}",
            base64Floats(1f, 2f, 3f));
    OpenAiCompatClient.injectHttpClient(client, mockHttpClient(200, body));

    List<VectorOrError> results = client.embed(List.of("hello"), floatContext());

    assertEquals(1, results.size());
    assertTrue(results.get(0).vector.isPresent());
    assertEquals(3, results.get(0).vector.get().numDimensions());
  }

  @Test
  public void embed_emptyInputsReturnEmptyInputError() throws Exception {
    OpenAiCompatClient client = newClient(openAiModel(Optional.empty()));
    OpenAiCompatClient.injectHttpClient(client, mockHttpClient(200, "{\"data\":[]}"));

    List<VectorOrError> results = client.embed(List.of("", ""), floatContext());

    assertEquals(2, results.size());
    assertEquals(VectorOrError.EMPTY_INPUT_ERROR, results.get(0));
    assertEquals(VectorOrError.EMPTY_INPUT_ERROR, results.get(1));
  }

  @Test
  public void embed_invalidRequestFailsFastAsPerInputError() throws Exception {
    OpenAiCompatClient client = newClient(openAiModel(Optional.empty()));
    OpenAiCompatClient.injectHttpClient(
        client, mockHttpClient(400, "{\"error\":{\"message\":\"bad model\"}}"));

    List<VectorOrError> results = client.embed(List.of("hello"), floatContext());

    assertEquals(1, results.size());
    assertTrue(results.get(0).errorMessage.isPresent());
    assertTrue(results.get(0).vector.isEmpty());
  }

  @Test
  public void embed_nonFloatQuantizationFailsFast() throws Exception {
    OpenAiCompatClient client = newClient(openAiModel(Optional.empty()));
    OpenAiCompatClient.injectHttpClient(client, mockHttpClient(200, "{\"data\":[]}"));

    EmbeddingRequestContext scalarContext =
        new EmbeddingRequestContext(
            "testdb", "testIndex", "testCollection", 3, VectorAutoEmbedQuantization.SCALAR);

    org.junit.Assert.assertThrows(
        EmbeddingProviderNonTransientException.class,
        () -> client.embed(List.of("hello"), scalarContext));
  }

  @Test
  public void embed_attachesBearerHeaderOnlyWhenApiKeyPresent() throws Exception {
    String body =
        String.format("{\"data\":[{\"embedding\":\"%s\",\"index\":0}]}", base64Floats(1f, 2f, 3f));

    // With API key -> Authorization header present.
    OpenAiCompatClient withKey = newClient(openAiModel(Optional.of("secret-key")));
    HttpClient withKeyHttp = mockHttpClient(200, body);
    OpenAiCompatClient.injectHttpClient(withKey, withKeyHttp);
    withKey.embed(List.of("hello"), floatContext());
    HttpRequest withKeyRequest = captureRequest(withKeyHttp);
    assertTrue(withKeyRequest.headers().firstValue("Authorization").isPresent());
    assertEquals(
        "Bearer secret-key", withKeyRequest.headers().firstValue("Authorization").get());

    // Without API key -> no Authorization header (keyless local engine).
    OpenAiCompatClient withoutKey = newClient(openAiModel(Optional.empty()));
    HttpClient withoutKeyHttp = mockHttpClient(200, body);
    OpenAiCompatClient.injectHttpClient(withoutKey, withoutKeyHttp);
    withoutKey.embed(List.of("hello"), floatContext());
    HttpRequest withoutKeyRequest = captureRequest(withoutKeyHttp);
    assertFalse(withoutKeyRequest.headers().firstValue("Authorization").isPresent());
  }

  @Test
  public void embed_usesAzureApiKeyHeaderWhenConfigured() throws Exception {
    String body =
        String.format("{\"data\":[{\"embedding\":\"%s\",\"index\":0}]}", base64Floats(1f, 2f, 3f));

    // authHeaderName=api-key (Azure OpenAI) -> raw key in api-key header, no Bearer, no
    // Authorization header.
    OpenAiCompatClient client =
        newClient(openAiModel(Optional.of("secret-key"), Optional.of("api-key")));
    HttpClient http = mockHttpClient(200, body);
    OpenAiCompatClient.injectHttpClient(client, http);
    client.embed(List.of("hello"), floatContext());
    HttpRequest request = captureRequest(http);
    assertEquals("secret-key", request.headers().firstValue("api-key").orElse(null));
    assertFalse(request.headers().firstValue("Authorization").isPresent());
  }

  @Test
  public void embed_forwardsDimensionsOnlyWhenOptedIn() throws Exception {
    String body =
        String.format("{\"data\":[{\"embedding\":\"%s\",\"index\":0}]}", base64Floats(1f, 2f, 3f));

    // forwardDimensions=true with a configured outputDimensions -> `dimensions` in request body.
    EmbeddingServiceConfig.OpenAiModelConfig withForward =
        new EmbeddingServiceConfig.OpenAiModelConfig(
            Optional.of(512), Optional.of(96), Optional.of(120_000), Optional.empty(),
            Optional.of(true));
    OpenAiCompatClient forwardClient =
        newClient(openAiModel(Optional.empty(), Optional.empty(), withForward));
    HttpClient forwardHttp = mockHttpClient(200, body);
    OpenAiCompatClient.injectHttpClient(forwardClient, forwardHttp);
    forwardClient.embed(List.of("hello"), floatContext());
    String forwardBody = requestBody(captureRequest(forwardHttp));
    assertTrue(
        "Expected dimensions forwarded when opted in: " + forwardBody,
        forwardBody.contains("\"dimensions\"") && forwardBody.contains("512"));

    // Default (no forwardDimensions): outputDimensions set but NOT forwarded (local engines reject).
    EmbeddingServiceConfig.OpenAiModelConfig noForward =
        new EmbeddingServiceConfig.OpenAiModelConfig(
            Optional.of(768), Optional.of(96), Optional.of(120_000), Optional.empty());
    OpenAiCompatClient defaultClient =
        newClient(openAiModel(Optional.empty(), Optional.empty(), noForward));
    HttpClient defaultHttp = mockHttpClient(200, body);
    OpenAiCompatClient.injectHttpClient(defaultClient, defaultHttp);
    defaultClient.embed(List.of("hello"), floatContext());
    String defaultBody = requestBody(captureRequest(defaultHttp));
    assertFalse(
        "Expected no dimensions field by default: " + defaultBody,
        defaultBody.contains("\"dimensions\""));
  }

  @Test
  public void embed_rateLimited429_throwsTransient() throws Exception {
    OpenAiCompatClient client = newClient(openAiModel(Optional.empty()));
    OpenAiCompatClient.injectHttpClient(client, mockHttpClient(429, "{\"error\":\"rate limit\"}"));

    org.junit.Assert.assertThrows(
        EmbeddingProviderTransientException.class,
        () -> client.embed(List.of("hello"), floatContext()));
  }

  @Test
  public void embed_timeout408_throwsTransient() throws Exception {
    OpenAiCompatClient client = newClient(openAiModel(Optional.empty()));
    OpenAiCompatClient.injectHttpClient(client, mockHttpClient(408, "{\"error\":\"timeout\"}"));

    org.junit.Assert.assertThrows(
        EmbeddingProviderTransientException.class,
        () -> client.embed(List.of("hello"), floatContext()));
  }

  @Test
  public void embed_serverError500_throwsTransient() throws Exception {
    OpenAiCompatClient client = newClient(openAiModel(Optional.empty()));
    OpenAiCompatClient.injectHttpClient(client, mockHttpClient(500, "internal error"));

    org.junit.Assert.assertThrows(
        EmbeddingProviderTransientException.class,
        () -> client.embed(List.of("hello"), floatContext()));
  }

  @Test
  public void embed_fewerVectorsThanInputs_throwsTransient() throws Exception {
    OpenAiCompatClient client = newClient(openAiModel(Optional.empty()));
    // Two non-empty inputs but the engine returns a single embedding.
    String body =
        String.format("{\"data\":[{\"embedding\":\"%s\",\"index\":0}]}", base64Floats(1f, 2f, 3f));
    OpenAiCompatClient.injectHttpClient(client, mockHttpClient(200, body));

    org.junit.Assert.assertThrows(
        EmbeddingProviderTransientException.class,
        () -> client.embed(List.of("hello", "world"), floatContext()));
  }

  private static HttpRequest captureRequest(HttpClient httpClient) throws Exception {
    ArgumentCaptor<HttpRequest> captor = ArgumentCaptor.forClass(HttpRequest.class);
    org.mockito.Mockito.verify(httpClient)
        .send(captor.capture(), any(HttpResponse.BodyHandler.class));
    return captor.getValue();
  }

  /** Reads the JSON body out of the request's {@link HttpRequest.BodyPublisher}. */
  private static String requestBody(HttpRequest request) throws Exception {
    HttpRequest.BodyPublisher publisher = request.bodyPublisher().orElseThrow();
    StringBuilder sb = new StringBuilder();
    CountDownLatch latch = new CountDownLatch(1);
    publisher.subscribe(
        new Flow.Subscriber<ByteBuffer>() {
          @Override
          public void onSubscribe(Flow.Subscription subscription) {
            subscription.request(Long.MAX_VALUE);
          }

          @Override
          public void onNext(ByteBuffer item) {
            sb.append(StandardCharsets.UTF_8.decode(item));
          }

          @Override
          public void onError(Throwable throwable) {
            latch.countDown();
          }

          @Override
          public void onComplete() {
            latch.countDown();
          }
        });
    latch.await(5, TimeUnit.SECONDS);
    return sb.toString();
  }
}
